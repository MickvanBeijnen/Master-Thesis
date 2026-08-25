import org.apache.lucene.analysis.Analyzer;
import org.apache.lucene.analysis.LowerCaseFilter;
import org.apache.lucene.analysis.StopFilter;
import org.apache.lucene.analysis.TokenFilter;
import org.apache.lucene.analysis.TokenStream;
import org.apache.lucene.analysis.Tokenizer;
import org.apache.lucene.analysis.CharArraySet;
import org.apache.lucene.analysis.en.EnglishAnalyzer;
import org.apache.lucene.analysis.en.PorterStemFilter;
import org.apache.lucene.analysis.pattern.PatternTokenizer;
import org.apache.lucene.analysis.tokenattributes.CharTermAttribute;
import org.apache.lucene.analysis.tokenattributes.KeywordAttribute;

import java.io.IOException;
import java.util.Set;
import java.util.regex.Pattern;

/**
 * CustomRegexAnalyzer
 *
 * A Lucene Analyzer that replicates Anserini's DefaultEnglishAnalyzer pipeline
 * but replaces StandardTokenizer with a PatternTokenizer using the same regex
 * as the custom Python preprocessing pipeline:
 *
 *     [a-z0-9]+(?:[+#&.-]+[a-z0-9]*)*|[!]
 *
 * This keeps compound terms like c++, covid-19, at&t, e-mail as single tokens,
 * which StandardTokenizer would otherwise split or strip.
 *
 * Pipeline:
 *   PatternTokenizer        — regex tokenization (same as custom Python pipeline)
 *   LowerCaseFilter         — identical to Anserini
 *   TrailingApostropheFilter— strips trailing apostrophes (straight and curly)
 *   SymbolKeywordFilter     — marks symbol-containing tokens to bypass stemming
 *   StopFilter              — identical to Anserini (EnglishAnalyzer stop words)
 *   PorterStemFilter        — identical to Anserini
 *   TrailingApostropheFilter— strips apostrophes that survive stemming
 *
 * Usage:
 *   javac -proc:none -cp "anserini-*-fatjar.jar" CustomRegexAnalyzer.java
 *
 * In Python via pyjnius:
 *   JCustomRegexAnalyzer = autoclass("CustomRegexAnalyzer")
 *   JAnalyzerUtils = autoclass("io.anserini.analysis.AnalyzerUtils")
 *   analyzer = JCustomRegexAnalyzer()
 *   tokens = [str(t) for t in JAnalyzerUtils.analyze(analyzer, text)]
 */
public class CustomRegexAnalyzer extends Analyzer {

    private static final CharArraySet STOP_WORDS =
        EnglishAnalyzer.ENGLISH_STOP_WORDS_SET;

    // group=0 returns the whole match as the token.
    // CASE_INSENSITIVE so LowerCaseFilter handles normalisation consistently.

    /** symbol: keeps compound terms like c++, covid-19, at&t */
    private static final Pattern PATTERN_SYMBOL = Pattern.compile(
        "[a-z0-9]+(?:[+#&.-]+[a-z0-9]*)*|[!]",
        Pattern.CASE_INSENSITIVE
    );

    /** alphanum: alphanumeric tokens only, no symbols */
    private static final Pattern PATTERN_ALPHANUM = Pattern.compile(
        "[a-z0-9]+",
        Pattern.CASE_INSENSITIVE
    );

    private final Pattern tokenPattern;

    /** Default constructor uses symbol mode. */
    public CustomRegexAnalyzer() {
        this("symbol");
    }

    /** @param mode "symbol" or "alphanum" */
    public CustomRegexAnalyzer(String mode) {
        switch (mode.toLowerCase()) {
            case "symbol":
                this.tokenPattern = PATTERN_SYMBOL;
                break;
            case "alphanum":
                this.tokenPattern = PATTERN_ALPHANUM;
                break;
            default:
                throw new IllegalArgumentException(
                    "Unknown mode '" + mode + "'. Choose: symbol | alphanum");
        }
    }

    @Override
    protected TokenStreamComponents createComponents(String fieldName) {
        Tokenizer   source = new PatternTokenizer(tokenPattern, 0);
        TokenStream result = new LowerCaseFilter(source);
        result = new TrailingApostropheFilter(result);
        result = new SymbolKeywordFilter(result);
        result = new StopFilter(result, STOP_WORDS);
        result = new PorterStemFilter(result);
        result = new TrailingApostropheFilter(result);
        return new TokenStreamComponents(source, result);
    }

    // =========================================================================
    // TrailingApostropheFilter
    //
    // Strips trailing straight (') and curly (\u2019\u2018) apostrophe characters
    // from tokens. Tokens that become empty after stripping are discarded.
    // =========================================================================

    private static final class TrailingApostropheFilter extends TokenFilter {

        private final CharTermAttribute termAtt =
            addAttribute(CharTermAttribute.class);

        TrailingApostropheFilter(TokenStream input) {
            super(input);
        }

        @Override
        public boolean incrementToken() throws IOException {
            while (input.incrementToken()) {
                String term = termAtt.toString();
                int end = term.length();
                while (end > 0) {
                    char c = term.charAt(end - 1);
                    if (c == '\'' || c == '\u2019' || c == '\u2018') {
                        end--;
                    } else {
                        break;
                    }
                }
                if (end == 0) continue;
                if (end < term.length()) {
                    termAtt.setEmpty().append(term, 0, end);
                }
                return true;
            }
            return false;
        }
    }

    // =========================================================================
    // SymbolKeywordFilter
    //
    // Marks tokens containing target symbols as keywords so PorterStemFilter
    // skips them. This preserves tokens like c++, e-mail, covid-19 unstemmed,
    // since running them through the stemmer would strip the symbols anyway.
    // =========================================================================

    private static final class SymbolKeywordFilter extends TokenFilter {

        private static final Set<Character> MERGE_SYMBOLS = Set.of(
            '+', '#', '&', '.', '-'
        );

        private final CharTermAttribute termAtt    = addAttribute(CharTermAttribute.class);
        private final KeywordAttribute  keywordAtt = addAttribute(KeywordAttribute.class);

        SymbolKeywordFilter(TokenStream input) {
            super(input);
        }

        @Override
        public boolean incrementToken() throws IOException {
            if (!input.incrementToken()) return false;
            String term = termAtt.toString();
            for (int i = 0; i < term.length(); i++) {
                if (MERGE_SYMBOLS.contains(term.charAt(i))) {
                    keywordAtt.setKeyword(true);
                    break;
                }
            }
            return true;
        }
    }
}