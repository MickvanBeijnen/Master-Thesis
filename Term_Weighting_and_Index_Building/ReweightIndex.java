import org.apache.lucene.analysis.core.WhitespaceAnalyzer;
import org.apache.lucene.document.BinaryDocValuesField;
import org.apache.lucene.document.Document;
import org.apache.lucene.document.Field;
import org.apache.lucene.document.FieldType;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.IndexOptions;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.index.IndexWriterConfig;
import org.apache.lucene.index.IndexableField;
import org.apache.lucene.index.LeafReader;
import org.apache.lucene.index.LeafReaderContext;
import org.apache.lucene.index.Terms;
import org.apache.lucene.index.TermsEnum;
import org.apache.lucene.store.FSDirectory;
import org.apache.lucene.util.Bits;
import org.apache.lucene.util.BytesRef;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;

/**
 * ReweightIndex
 *
 * Builds Lucene/Anserini indices with reweighted term frequencies.
 *
 * Modes:
 *   [DEBUG] constant         — every TF set to CONSTANT_TF
 *   [DEBUG] random           — every TF randomised in [1, RANDOM_MAX]
 *   [DEBUG] weighted_random  — existing TF scaled by random float
 *   These debug modes are for pipeline verification only and produce
 *   no meaningful term weights.
 *   rlm              — new_tf = α·tf + (1−α)·rlm_weight
 *   rlm_multi        — same as rlm, but builds all alpha variants in one pass
 *                      alphas: comma-separated, e.g. "1.0,0.75,0.5,0.25,0.0"
 *   cg               — new_tf = α·tf + (1−α)·cg_weight   (click-graph only)
 *   cg_multi         — same as cg, but builds all alpha variants in one pass
 *   combined         — new_tf = α·tf + β·rlm_weight + (1−α−β)·cg_weight
 *                      requires α + β ≤ 1
 *   combined_multi   — same as combined, cartesian product of alphas × betas
 *                      invalid pairs where α + β > 1 are skipped automatically
 *
 * Usage:
 *   javac -proc:none -cp "anserini-*-fatjar.jar" ReweightIndex.java
 *
 *   constant / random / weighted_random:
 *   java -cp ".:anserini-*-fatjar.jar" ReweightIndex \
 *        <input_index> <output_dir> <tokenizer> <mode> <max_val>
 *
 *   rlm:
 *   java -cp ".:anserini-*-fatjar.jar" ReweightIndex \
 *        <input_index> <output_dir> <tokenizer> rlm <max_val> <weights_tsv> <alpha>
 *
 *   rlm_multi:
 *   java -cp ".:anserini-*-fatjar.jar" ReweightIndex \
 *        <input_index> <output_dir> <tokenizer> rlm_multi <max_val> <weights_tsv> \
 *        <alpha1,alpha2,...>
 *
 *   cg:
 *   java -cp ".:anserini-*-fatjar.jar" ReweightIndex \
 *        <input_index> <output_dir> <tokenizer> cg <max_val> <weights_tsv> <alpha>
 *
 *   cg_multi:
 *   java -cp ".:anserini-*-fatjar.jar" ReweightIndex \
 *        <input_index> <output_dir> <tokenizer> cg_multi <max_val> <weights_tsv> \
 *        <alpha1,alpha2,...>
 *
 *   combined:
 *   java -cp ".:anserini-*-fatjar.jar" ReweightIndex \
 *        <input_index> <output_dir> <tokenizer> combined <max_val> \
 *        <rlm_weights_tsv> <cg_weights_tsv> <alpha> <beta>
 *
 *   combined_multi:
 *   java -cp ".:anserini-*-fatjar.jar" ReweightIndex \
 *        <input_index> <output_dir> <tokenizer> combined_multi <max_val> \
 *        <rlm_weights_tsv> <cg_weights_tsv> \
 *        <alpha1,alpha2,...> <beta1,beta2,...>
 */
public class ReweightIndex {

    // ── Debug mode constants (constant / random / weighted_random) ─────────
    // Not used in production — retained for pipeline verification only.
    private static final int   CONSTANT_TF = 1;
    private static final int   RANDOM_MAX  = 3;
    private static final float WEIGHT_MIN  = 0.5f;
    private static final float WEIGHT_MAX  = 1.5f;

    private static final String CONTENTS_FIELD = "contents";
    private static final String ID_FIELD       = "id";

    private static final FieldType CONTENTS_TYPE;
    static {
        CONTENTS_TYPE = new FieldType();
        CONTENTS_TYPE.setIndexOptions(IndexOptions.DOCS_AND_FREQS_AND_POSITIONS);
        CONTENTS_TYPE.setTokenized(true);
        CONTENTS_TYPE.setStored(true);
        CONTENTS_TYPE.setStoreTermVectors(true);
        CONTENTS_TYPE.setStoreTermVectorPositions(true);
        CONTENTS_TYPE.freeze();
    }

    // ── Small helper to represent an (alpha, beta) pair ──────────────────────
    private static class AlphaBeta {
        final float alpha;
        final float beta;
        AlphaBeta(float alpha, float beta) {
            this.alpha = alpha;
            this.beta  = beta;
        }
        @Override public String toString() {
            return String.format("(α=%.2f, β=%.2f)", alpha, beta);
        }
    }

    public static void main(String[] args) throws IOException {

        if (args.length < 5) {
            printUsage();
            System.exit(1);
        }

        String inputPath       = args[0];
        String outputDir       = args[1];
        String tokenizerSuffix = args[2];
        String mode            = args[3].toLowerCase();
        float  maxVal          = Float.parseFloat(args[4]);  // for rlm/cg single modes
        float  maxValRlm       = maxVal;  // overridden in combined modes
        float  maxValCg        = maxVal;  // overridden in combined modes

        Random rng = new Random(42);

        // ── Parse mode-specific arguments ────────────────────────────────────
        List<Float>     rlmAlphas  = new ArrayList<>();  // for rlm / rlm_multi
        List<Float>     cgAlphas   = new ArrayList<>();  // for cg / cg_multi
        List<AlphaBeta> abPairs    = new ArrayList<>();  // for combined / combined_multi
        String          rlmTsv    = null;
        String          cgTsv     = null;

        switch (mode) {
            case "rlm":
                requireArgs(args, 7, "rlm <max_val> <weights_tsv> <alpha>");
                rlmTsv = args[5];
                rlmAlphas.add(Float.parseFloat(args[6]));
                break;

            case "rlm_multi":
                requireArgs(args, 7, "rlm_multi <max_val> <weights_tsv> <alpha1,alpha2,...>");
                rlmTsv = args[5];
                for (String a : args[6].split(","))
                    rlmAlphas.add(Float.parseFloat(a.trim()));
                break;

            case "cg":
                requireArgs(args, 7, "cg <max_val> <weights_tsv> <alpha>");
                cgTsv = args[5];
                cgAlphas.add(Float.parseFloat(args[6]));
                break;

            case "cg_multi":
                requireArgs(args, 7, "cg_multi <max_val> <weights_tsv> <alpha1,alpha2,...>");
                cgTsv = args[5];
                for (String a : args[6].split(","))
                    cgAlphas.add(Float.parseFloat(a.trim()));
                break;

            case "combined":
                requireArgs(args, 10, "combined <max_val_rlm> <max_val_cg> <rlm_tsv> <cg_tsv> <alpha> <beta>");
                maxValRlm = Float.parseFloat(args[4]);
                maxValCg  = Float.parseFloat(args[5]);
                rlmTsv = args[6];
                cgTsv  = args[7];
                float a = Float.parseFloat(args[8]);
                float b = Float.parseFloat(args[9]);
                if (a + b > 1.0f + 1e-6f) {
                    System.err.println("Error: alpha + beta must be <= 1.0 (got " + (a+b) + ")");
                    System.exit(1);
                }
                abPairs.add(new AlphaBeta(a, b));
                break;

            case "combined_multi":
                requireArgs(args, 10, "combined_multi <max_val_rlm> <max_val_cg> <rlm_tsv> <cg_tsv> <alphas> <betas>");
                maxValRlm = Float.parseFloat(args[4]);
                maxValCg  = Float.parseFloat(args[5]);
                rlmTsv = args[6];
                cgTsv  = args[7];
                List<Float> alphaList = new ArrayList<>();
                List<Float> betaList  = new ArrayList<>();
                for (String av : args[8].split(",")) alphaList.add(Float.parseFloat(av.trim()));
                for (String bv : args[9].split(",")) betaList.add(Float.parseFloat(bv.trim()));
                int skipped = 0;
                for (float alpha : alphaList) {
                    for (float beta : betaList) {
                        if (alpha + beta <= 1.0f + 1e-6f) {
                            abPairs.add(new AlphaBeta(alpha, beta));
                        } else {
                            skipped++;
                        }
                    }
                }
                System.out.println("Combined pairs: " + abPairs.size() + " valid, " + skipped + " skipped (α+β>1)");
                break;

            case "constant":
            case "random":
            case "weighted_random":
                // Debug modes: pipeline verification only, not for actual experiments
                break;

            default:
                System.err.println("Unknown mode: " + mode);
                printUsage();
                System.exit(1);
        }

        System.out.println("Input index : " + inputPath);
        System.out.println("Output dir  : " + outputDir);
        System.out.println("Tokenizer   : " + tokenizerSuffix);
        System.out.println("Mode        : " + mode);
        if (mode.equals("combined") || mode.equals("combined_multi")) {
            System.out.println("MAX_VAL RLM : " + maxValRlm);
            System.out.println("MAX_VAL CG  : " + maxValCg);
        } else {
            System.out.println("MAX_VAL     : " + maxVal);
        }

        if (mode.equals("constant") || mode.equals("random") || mode.equals("weighted_random")) {
            System.out.println();
            System.out.println("WARNING: '" + mode + "' is a DEBUG mode intended for pipeline");
            System.out.println("         verification only. It produces no meaningful term");
            System.out.println("         weights and should not be used for actual experiments.");
            System.out.println();
        }

        // ── Scan weights files for probability ranges ─────────────────────────
        // RLM and CG weights are log-scaled into [1, MAX_VAL+1] using the same
        // formula, so we scan each file independently for its own [probMin, probMax].
        double rlmLogProbMin = 0.0, rlmLogRange = 1.0;
        double cgLogProbMin  = 0.0, cgLogRange  = 1.0;

        boolean needsRlm = mode.equals("rlm") || mode.equals("rlm_multi")
                        || mode.equals("combined") || mode.equals("combined_multi");
        boolean needsCg  = mode.equals("cg") || mode.equals("cg_multi")
                        || mode.equals("combined") || mode.equals("combined_multi");

        if (needsRlm) {
            System.out.println("Scanning RLM weights for probability range...");
            double[] range = scanProbRange(rlmTsv);
            rlmLogProbMin = range[0];
            rlmLogRange   = range[1];
            System.out.println("  RLM log range: [" + Math.exp(rlmLogProbMin)
                + ", " + Math.exp(rlmLogProbMin + rlmLogRange) + "]");
        }
        if (needsCg) {
            System.out.println("Scanning CG weights for probability range...");
            double[] range = scanProbRange(cgTsv);
            cgLogProbMin = range[0];
            cgLogRange   = range[1];
            System.out.println("  CG log range: [" + Math.exp(cgLogProbMin)
                + ", " + Math.exp(cgLogProbMin + cgLogRange) + "]");
        }

        // Whether this run combines both signals (controls scoring formula below)
        boolean isCombinedMode = mode.equals("combined") || mode.equals("combined_multi");
        boolean isRlmOnlyMode  = mode.equals("rlm") || mode.equals("rlm_multi");
        boolean isCgOnlyMode   = mode.equals("cg")  || mode.equals("cg_multi");

        // ── Open index writers ────────────────────────────────────────────────
        // Key for rlm/rlm_multi:       "a<alpha>"               (rlm-only)
        // Key for cg/cg_multi:         "a<alpha>"                (cg-only)
        // Key for combined variants:   "a<alpha>_b<beta>"
        // Key for simple modes:        "simple"
        Map<String, IndexWriter> writers = new HashMap<>();

        if (isRlmOnlyMode) {
            for (float alpha : rlmAlphas) {
                String key     = alphaKey(alpha);
                String outPath = outputDir + "/index_rlm_" + tokenizerSuffix
                    + "_maxval" + fmtVal(maxVal)
                    + "_alpha_" + fmt(alpha);
                writers.put(key, openWriter(outPath));
                System.out.println("  Writer: alpha=" + alpha + " → " + outPath);
            }
        } else if (isCgOnlyMode) {
            for (float alpha : cgAlphas) {
                String key     = alphaKey(alpha);
                String outPath = outputDir + "/index_cg_" + tokenizerSuffix
                    + "_maxval" + fmtVal(maxVal)
                    + "_alpha_" + fmt(alpha);
                writers.put(key, openWriter(outPath));
                System.out.println("  Writer: alpha=" + alpha + " → " + outPath);
            }
        } else if (isCombinedMode) {
            for (AlphaBeta ab : abPairs) {
                String key     = abKey(ab);
                String outPath = outputDir + "/index_combined_" + tokenizerSuffix
                    + "_maxvalrlm" + fmtVal(maxValRlm)
                    + "_maxvalcg"  + fmtVal(maxValCg)
                    + "_a" + fmt(ab.alpha)
                    + "_b" + fmt(ab.beta);
                writers.put(key, openWriter(outPath));
                System.out.println("  Writer: " + ab + " → " + outPath);
            }
        } else {
            // constant / random / weighted_random
            String outPath = outputDir + "/index_" + mode + "_" + tokenizerSuffix;
            writers.put("simple", openWriter(outPath));
            System.out.println("  Writer → " + outPath);
        }

        // ── Open source index + weight stream(s) ─────────────────────────────
        FSDirectory    srcDir = FSDirectory.open(Paths.get(inputPath));
        DirectoryReader reader = DirectoryReader.open(srcDir);

        BufferedReader rlmReader = null;
        BufferedReader cgReader  = null;
        // Single-element array wrappers so lambdas in collectWeights can
        // write back the updated buffer reference (lambdas can't assign to
        // plain local variables).
        String[][]     rlmBuf   = {null};
        String[][]     cgBuf    = {null};

        if (needsRlm) {
            rlmReader = new BufferedReader(new FileReader(rlmTsv));
            rlmReader.readLine(); // header
            String first = rlmReader.readLine();
            if (first != null) rlmBuf[0] = first.split("\t");
        }
        if (needsCg) {
            cgReader = new BufferedReader(new FileReader(cgTsv));
            cgReader.readLine(); // header
            String first = cgReader.readLine();
            if (first != null) cgBuf[0] = first.split("\t");
        }

        // ── Main rewrite loop ─────────────────────────────────────────────────
        int numDocs      = reader.maxDoc();
        int numRewritten = 0;

        System.out.println("Total documents: " + numDocs);
        System.out.println("Pass 2: streaming weights during indexing...");

        for (LeafReaderContext ctx : reader.leaves()) {
            LeafReader leaf    = ctx.reader();
            int        docBase = ctx.docBase;
            Bits       live    = leaf.getLiveDocs();

            for (int segDocId = 0; segDocId < leaf.maxDoc(); segDocId++) {

                if (live != null && !live.get(segDocId)) continue;

                int      docId     = docBase + segDocId;
                Document storedDoc = reader.storedFields().document(docId);
                Terms    terms     = leaf.termVectors().get(segDocId, CONTENTS_FIELD);

                // Advance weight streams to current docId
                if (needsRlm) rlmBuf[0] = advanceTo(rlmReader, rlmBuf[0], docId);
                if (needsCg)  cgBuf[0]  = advanceTo(cgReader,  cgBuf[0],  docId);

                // Collect weights for this document from each stream
                Map<String, Double> rlmWeights = needsRlm
                    ? collectWeights(rlmReader, rlmBuf[0], docId, result -> rlmBuf[0] = result)
                    : new HashMap<>();
                Map<String, Double> cgWeights  = needsCg
                    ? collectWeights(cgReader,  cgBuf[0],  docId, result -> cgBuf[0]  = result)
                    : new HashMap<>();

                // Docs with no term vector: copy as-is to all writers
                if (terms == null) {
                    for (IndexWriter w : writers.values()) w.addDocument(storedDoc);
                    continue;
                }

                // Build synthetic TF maps keyed by writer key
                Map<String, Map<String, Integer>> tfByKey = new HashMap<>();
                for (String key : writers.keySet())
                    tfByKey.put(key, new LinkedHashMap<>());

                TermsEnum termsEnum = terms.iterator();
                BytesRef  termBytes;

                while ((termBytes = termsEnum.next()) != null) {
                    String term   = termBytes.utf8ToString();
                    long   origTf = termsEnum.totalTermFreq();

                    if (mode.equals("constant")) {
                        tfByKey.get("simple").put(term, CONSTANT_TF);

                    } else if (mode.equals("random")) {
                        tfByKey.get("simple").put(term, 1 + rng.nextInt(RANDOM_MAX));

                    } else if (mode.equals("weighted_random")) {
                        float scale = WEIGHT_MIN + rng.nextFloat() * (WEIGHT_MAX - WEIGHT_MIN);
                        tfByKey.get("simple").put(term, Math.max(1, Math.round(origTf * scale)));

                    } else if (isRlmOnlyMode) {
                        // rlm / rlm_multi: new_tf = α·tf + (1−α)·rlm
                        int rlmScaled = scaleWeight(rlmWeights, term, rlmLogProbMin, rlmLogRange, maxVal);
                        for (float alpha : rlmAlphas) {
                            int newTf;
                            if (rlmScaled >= 0) {
                                newTf = (int) Math.max(1, Math.round(
                                    alpha * origTf + (1.0 - alpha) * rlmScaled));
                            } else {
                                // fallback: no RLM weight → keep original TF
                                newTf = (int) Math.max(1, origTf);
                            }
                            tfByKey.get(alphaKey(alpha)).put(term, newTf);
                        }

                    } else if (isCgOnlyMode) {
                        // cg / cg_multi: new_tf = α·tf + (1−α)·cg
                        int cgScaled = scaleWeight(cgWeights, term, cgLogProbMin, cgLogRange, maxVal);
                        for (float alpha : cgAlphas) {
                            int newTf;
                            if (cgScaled >= 0) {
                                newTf = (int) Math.max(1, Math.round(
                                    alpha * origTf + (1.0 - alpha) * cgScaled));
                            } else {
                                // fallback: no CG weight → keep original TF
                                newTf = (int) Math.max(1, origTf);
                            }
                            tfByKey.get(alphaKey(alpha)).put(term, newTf);
                        }

                    } else if (isCombinedMode) {
                        // combined / combined_multi: new_tf = α·tf + β·rlm + (1−α−β)·cg
                        int rlmScaled = scaleWeight(rlmWeights, term, rlmLogProbMin, rlmLogRange, maxValRlm);
                        int cgScaled  = scaleWeight(cgWeights,  term, cgLogProbMin,  cgLogRange,  maxValCg);
                        for (AlphaBeta ab : abPairs) {
                            int newTf;
                            double tfComponent  = ab.alpha * origTf;
                            // For each missing signal, fall back to original TF
                            // scaled by its coefficient so the total stays balanced
                            double rlmComponent = (rlmScaled >= 0)
                                ? ab.beta * rlmScaled
                                : ab.beta * origTf;
                            double cgComponent  = (cgScaled >= 0)
                                ? (1.0 - ab.alpha - ab.beta) * cgScaled
                                : (1.0 - ab.alpha - ab.beta) * origTf;
                            newTf = (int) Math.max(1, Math.round(
                                tfComponent + rlmComponent + cgComponent));
                            tfByKey.get(abKey(ab)).put(term, newTf);
                        }
                    }
                }

                // Write to all index writers
                for (Map.Entry<String, IndexWriter> entry : writers.entrySet()) {
                    Document doc = buildDocument(storedDoc, tfByKey.get(entry.getKey()));
                    entry.getValue().addDocument(doc);
                }

                numRewritten++;
                if (numRewritten % 100_000 == 0)
                    System.out.println("  Rewritten: " + numRewritten + " / " + numDocs);
            }
        }

        // ── Cleanup ───────────────────────────────────────────────────────────
        if (rlmReader != null) rlmReader.close();
        if (cgReader  != null) cgReader.close();

        for (IndexWriter w : writers.values()) {
            w.forceMerge(1);
            w.close();
        }
        reader.close();
        srcDir.close();

        System.out.println("Done. Rewritten " + numRewritten + " / " + numDocs + " documents.");
        System.out.println("Built " + writers.size() + " index variant(s).");
    }

    // ── Helpers ───────────────────────────────────────────────────────────────

    /** Scan a weights TSV for [logProbMin, logRange]. Returns double[]{logMin, logRange}. */
    private static double[] scanProbRange(String tsvPath) throws IOException {
        double probMin = Double.MAX_VALUE;
        double probMax = -Double.MAX_VALUE;
        try (BufferedReader br = new BufferedReader(new FileReader(tsvPath))) {
            br.readLine(); // header
            String line;
            while ((line = br.readLine()) != null) {
                String[] p = line.split("\t");
                if (p.length < 4) continue;
                double prob = Double.parseDouble(p[3].trim());
                if (prob > 0) {
                    probMin = Math.min(probMin, prob);
                    probMax = Math.max(probMax, prob);
                }
            }
        }
        double logMin   = Math.log(probMin);
        double logRange = Math.log(probMax) - logMin;
        return new double[]{logMin, logRange};
    }

    /**
     * Advance a weight stream's buffered line to the first line with
     * traversal_idx >= targetDocId. Returns the updated buffer.
     */
    private static String[] advanceTo(BufferedReader br, String[] buf, int targetDocId)
            throws IOException {
        while (buf != null) {
            if (buf.length < 4) {
                String next = br.readLine();
                buf = (next != null) ? next.split("\t") : null;
                continue;
            }
            if (Integer.parseInt(buf[0].trim()) < targetDocId) {
                String next = br.readLine();
                buf = (next != null) ? next.split("\t") : null;
            } else {
                break;
            }
        }
        return buf;
    }

    /**
     * Functional interface so collectWeights can write back the updated buffer
     * reference via a lambda (Java lacks pass-by-reference for arrays).
     */
    @FunctionalInterface
    interface BufSetter { void set(String[] buf); }

    /**
     * Collect all (term → probability) entries for targetDocId from the stream.
     * Leaves the buffer pointing at the first line AFTER the current doc.
     * Writes the updated buffer back via setter.
     */
    private static Map<String, Double> collectWeights(
            BufferedReader br, String[] buf, int targetDocId, BufSetter setter)
            throws IOException {
        Map<String, Double> weights = new HashMap<>();
        while (buf != null) {
            if (buf.length < 4) {
                String next = br.readLine();
                buf = (next != null) ? next.split("\t") : null;
                continue;
            }
            int idx = Integer.parseInt(buf[0].trim());
            if (idx == targetDocId) {
                weights.put(buf[2].trim(), Double.parseDouble(buf[3].trim()));
                String next = br.readLine();
                buf = (next != null) ? next.split("\t") : null;
            } else {
                break;
            }
        }
        setter.set(buf);
        return weights;
    }

    /**
     * Log-scale a probability weight into [1, maxVal+1].
     * Returns -1 if the term has no entry in the weights map (missing signal).
     */
    private static int scaleWeight(Map<String, Double> weights, String term,
                                   double logProbMin, double logRange, float maxVal) {
        Double prob = weights.get(term);
        if (prob == null || prob <= 0) return -1;
        double logProb = Math.log(prob);
        double scaled  = (logRange > 0)
            ? (logProb - logProbMin) / logRange * maxVal
            : maxVal / 2.0;
        return (int) Math.round(scaled) + 1;
    }

    private static Document buildDocument(Document source, Map<String, Integer> syntheticTf) {
        Document doc = new Document();
        for (IndexableField field : source.getFields()) {
            if (!field.name().equals(CONTENTS_FIELD)) doc.add(field);
        }
        String idValue = source.get(ID_FIELD);
        if (idValue != null)
            doc.add(new BinaryDocValuesField(ID_FIELD, new BytesRef(idValue)));

        StringBuilder sb = new StringBuilder();
        for (Map.Entry<String, Integer> entry : syntheticTf.entrySet()) {
            String term  = entry.getKey();
            int    count = entry.getValue();
            for (int i = 0; i < count; i++) sb.append(term).append(' ');
        }
        doc.add(new Field(CONTENTS_FIELD, sb.toString().trim(), CONTENTS_TYPE));
        return doc;
    }

    private static IndexWriter openWriter(String path) throws IOException {
        FSDirectory dir = FSDirectory.open(Paths.get(path));
        IndexWriterConfig cfg = new IndexWriterConfig(new WhitespaceAnalyzer());
        cfg.setOpenMode(IndexWriterConfig.OpenMode.CREATE);
        return new IndexWriter(dir, cfg);
    }

    private static String alphaKey(float alpha) {
        return "a" + fmt(alpha);
    }

    private static String abKey(AlphaBeta ab) {
        return "a" + fmt(ab.alpha) + "_b" + fmt(ab.beta);
    }

    private static String fmt(float v) {
        return String.format("%.2f", v).replace(".", "_");
    }

    /** Format a maxVal float, dropping unnecessary trailing zeros.
     *  10.0 -> "10", 7.5 -> "7.5", 12.5 -> "12.5" */
    private static String fmtVal(float v) {
        if (v == Math.floor(v)) {
            return String.valueOf((int) v);
        }
        // Remove trailing zeros after decimal point
        return String.valueOf(v).replaceAll("0+$", "").replaceAll("\\.$", "");
    }

    private static void requireArgs(String[] args, int n, String usage) {
        if (args.length < n) {
            System.err.println("Usage: ReweightIndex <input> <output_dir> <tokenizer> " + usage);
            System.exit(1);
        }
    }

    private static void printUsage() {
        System.err.println("Usage: ReweightIndex <input_index> <output_dir> <tokenizer> <mode> <max_val> [mode-args...]");
        System.err.println("  Production modes: rlm | rlm_multi | cg | cg_multi | combined | combined_multi");
        System.err.println("  Debug modes only: constant | random | weighted_random");
    }
}