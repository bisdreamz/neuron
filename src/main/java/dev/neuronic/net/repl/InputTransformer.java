package dev.neuronic.net.repl;

/**
 * Transforms user input strings into model-ready inputs.
 * 
 * @param <T> the type of input the model expects (e.g., float[], String[], etc.)
 */
@FunctionalInterface
public interface InputTransformer<T> {
    /**
     * Transform a user input string into model input format.
     * 
     * @param input raw user input
     * @return transformed input ready for the model
     * @throws IllegalArgumentException if input cannot be transformed
     */
    T transform(String input);
}