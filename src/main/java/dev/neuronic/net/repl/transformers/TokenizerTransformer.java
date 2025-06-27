package dev.neuronic.net.repl.transformers;

import dev.neuronic.net.repl.InputTransformer;
import java.util.Arrays;

/**
 * Tokenizes text input by splitting on delimiters.
 * Commonly used for language models and text processing.
 */
public class TokenizerTransformer implements InputTransformer<String[]> {
    
    private final String delimiter;
    private final boolean toLowerCase;
    private final int maxTokens;
    
    /**
     * Create a tokenizer with custom settings.
     */
    public TokenizerTransformer(String delimiter, boolean toLowerCase, int maxTokens) {
        this.delimiter = delimiter;
        this.toLowerCase = toLowerCase;
        this.maxTokens = maxTokens;
    }
    
    /**
     * Create a tokenizer that splits on spaces.
     */
    public static TokenizerTransformer splitOnSpaces() {
        return new TokenizerTransformer(" ", true, Integer.MAX_VALUE);
    }
    
    /**
     * Create a tokenizer with custom delimiter.
     */
    public static TokenizerTransformer splitOn(String delimiter) {
        return new TokenizerTransformer(delimiter, true, Integer.MAX_VALUE);
    }
    
    /**
     * Create a tokenizer that preserves case.
     */
    public static TokenizerTransformer caseSensitive(String delimiter) {
        return new TokenizerTransformer(delimiter, false, Integer.MAX_VALUE);
    }
    
    @Override
    public String[] transform(String input) {
        if (input == null || input.trim().isEmpty()) {
            throw new IllegalArgumentException("Input cannot be empty");
        }
        
        String processed = toLowerCase ? input.toLowerCase() : input;
        String[] tokens = processed.split(delimiter);
        
        // Remove empty tokens
        tokens = Arrays.stream(tokens)
            .filter(s -> !s.trim().isEmpty())
            .toArray(String[]::new);
        
        if (tokens.length == 0) {
            throw new IllegalArgumentException("No valid tokens found in input");
        }
        
        // Limit tokens if specified
        if (tokens.length > maxTokens) {
            tokens = Arrays.copyOf(tokens, maxTokens);
        }
        
        return tokens;
    }
}