package dev.neuronic.net.repl.formatters;

import dev.neuronic.net.repl.OutputFormatter;
import java.util.Arrays;

/**
 * Formats classification outputs showing top predictions with probabilities.
 */
public class ClassificationFormatter implements OutputFormatter<Object> {
    
    private final String[] classNames;
    private final int topK;
    private final boolean showProbabilities;
    
    /**
     * Create a formatter with class names and settings.
     */
    public ClassificationFormatter(String[] classNames, int topK, boolean showProbabilities) {
        this.classNames = classNames;
        this.topK = topK;
        this.showProbabilities = showProbabilities;
    }
    
    /**
     * Create a basic formatter showing predicted class index.
     */
    public static ClassificationFormatter basic() {
        return new ClassificationFormatter(null, 1, false);
    }
    
    /**
     * Create a formatter with class names.
     */
    public static ClassificationFormatter withClassNames(String... names) {
        return new ClassificationFormatter(names, 1, true);
    }
    
    /**
     * Create a formatter showing top-k predictions.
     */
    public static ClassificationFormatter topK(int k) {
        return new ClassificationFormatter(null, k, true);
    }
    
    @Override
    public String format(Object output) {
        if (output instanceof Integer) {
            // Simple class index
            int classIdx = (Integer) output;
            if (classNames != null && classIdx < classNames.length) {
                return "Predicted: " + classNames[classIdx];
            } else {
                return "Predicted class: " + classIdx;
            }
        } else if (output instanceof float[]) {
            // Probability distribution
            float[] probs = (float[]) output;
            return formatProbabilities(probs);
        } else {
            return "Unknown output format: " + output.getClass().getSimpleName();
        }
    }
    
    private String formatProbabilities(float[] probs) {
        // Get top K indices
        Integer[] indices = new Integer[probs.length];
        for (int i = 0; i < probs.length; i++) {
            indices[i] = i;
        }
        
        Arrays.sort(indices, (a, b) -> Float.compare(probs[b], probs[a]));
        
        StringBuilder sb = new StringBuilder();
        sb.append("Predictions:\n");
        
        int count = Math.min(topK, probs.length);
        for (int i = 0; i < count; i++) {
            int idx = indices[i];
            String className = (classNames != null && idx < classNames.length) 
                ? classNames[idx] 
                : "Class " + idx;
            
            if (showProbabilities) {
                sb.append(String.format("  %s: %.2f%%\n", className, probs[idx] * 100));
            } else {
                sb.append("  ").append(className).append("\n");
            }
        }
        
        return sb.toString().trim();
    }
}