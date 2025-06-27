package dev.neuronic.net.activators;

import java.util.concurrent.ExecutorService;

public interface Activator {

    public void activate(float[] input, float[] output);

    public void derivative(float[] input, float[] output);
    
    // Executor version with smart default - auto-submits if not overridden
    default void activate(float[] input, float[] output, ExecutorService executor) {
        if (executor == null) {
            activate(input, output);
            return;
        }
        
        // Smart default: submit to executor and wait for completion
        try {
            executor.submit(() -> {
                activate(input, output);
                return null;
            }).get();
        } catch (Exception e) {
            throw new RuntimeException("Parallel activation failed", e);
        }
    }
    
    default void derivative(float[] input, float[] output, ExecutorService executor) {
        if (executor == null) {
            derivative(input, output);
            return;
        }
        
        // Smart default: submit to executor and wait for completion
        try {
            executor.submit(() -> {
                derivative(input, output);
                return null;
            }).get();
        } catch (Exception e) {
            throw new RuntimeException("Parallel derivative computation failed", e);
        }
    }

}
