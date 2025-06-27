package dev.neuronic.net.repl.formatters;

import dev.neuronic.net.repl.OutputFormatter;

/**
 * Formats regression outputs with configurable precision.
 */
public class RegressionFormatter implements OutputFormatter<Object> {
    
    private final int decimalPlaces;
    private final String[] outputNames;
    
    /**
     * Create a formatter with custom settings.
     */
    public RegressionFormatter(int decimalPlaces, String[] outputNames) {
        this.decimalPlaces = decimalPlaces;
        this.outputNames = outputNames;
    }
    
    /**
     * Create a basic formatter with 4 decimal places.
     */
    public static RegressionFormatter basic() {
        return new RegressionFormatter(4, null);
    }
    
    /**
     * Create a formatter with custom precision.
     */
    public static RegressionFormatter withPrecision(int decimalPlaces) {
        return new RegressionFormatter(decimalPlaces, null);
    }
    
    /**
     * Create a formatter with named outputs.
     */
    public static RegressionFormatter withNames(String... names) {
        return new RegressionFormatter(4, names);
    }
    
    @Override
    public String format(Object output) {
        if (output instanceof Float) {
            return formatSingle((Float) output, 0);
        } else if (output instanceof float[]) {
            return formatArray((float[]) output);
        } else {
            return "Unknown output format: " + output.getClass().getSimpleName();
        }
    }
    
    private String formatSingle(float value, int index) {
        String format = "%." + decimalPlaces + "f";
        String formatted = String.format(format, value);
        
        if (outputNames != null && index < outputNames.length) {
            return outputNames[index] + ": " + formatted;
        } else {
            return "Output: " + formatted;
        }
    }
    
    private String formatArray(float[] values) {
        if (values.length == 1) {
            return formatSingle(values[0], 0);
        }
        
        StringBuilder sb = new StringBuilder();
        sb.append("Outputs:\n");
        
        String format = "  %s: %." + decimalPlaces + "f\n";
        
        for (int i = 0; i < values.length; i++) {
            String name = (outputNames != null && i < outputNames.length) 
                ? outputNames[i] 
                : "Output " + (i + 1);
            sb.append(String.format(format, name, values[i]));
        }
        
        return sb.toString().trim();
    }
}