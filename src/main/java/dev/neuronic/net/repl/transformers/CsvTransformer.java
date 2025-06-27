package dev.neuronic.net.repl.transformers;

import dev.neuronic.net.repl.InputTransformer;

/**
 * Transforms CSV input into float arrays.
 * Useful for regression and classification models.
 */
public class CsvTransformer implements InputTransformer<float[]> {
    
    private final String delimiter;
    private final int expectedColumns;
    
    /**
     * Create a CSV transformer with custom delimiter and column validation.
     */
    public CsvTransformer(String delimiter, int expectedColumns) {
        this.delimiter = delimiter;
        this.expectedColumns = expectedColumns;
    }
    
    /**
     * Create a CSV transformer for comma-separated values.
     */
    public static CsvTransformer commaSeparated() {
        return new CsvTransformer(",", -1);
    }
    
    /**
     * Create a CSV transformer with expected column count.
     */
    public static CsvTransformer withColumns(int columns) {
        return new CsvTransformer(",", columns);
    }
    
    /**
     * Create a CSV transformer with custom delimiter.
     */
    public static CsvTransformer withDelimiter(String delimiter) {
        return new CsvTransformer(delimiter, -1);
    }
    
    @Override
    public float[] transform(String input) {
        if (input == null || input.trim().isEmpty()) {
            throw new IllegalArgumentException("Input cannot be empty");
        }
        
        String[] parts = input.trim().split(delimiter);
        
        if (expectedColumns > 0 && parts.length != expectedColumns) {
            throw new IllegalArgumentException(
                String.format("Expected %d columns, got %d", expectedColumns, parts.length));
        }
        
        float[] values = new float[parts.length];
        
        for (int i = 0; i < parts.length; i++) {
            try {
                values[i] = Float.parseFloat(parts[i].trim());
            } catch (NumberFormatException e) {
                throw new IllegalArgumentException(
                    String.format("Invalid number at position %d: '%s'", i + 1, parts[i].trim()));
            }
        }
        
        return values;
    }
}