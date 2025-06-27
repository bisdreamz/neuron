package dev.neuronic.net.repl;

/**
 * Optional interface for models that can generate extended responses.
 * Primarily used for language models that generate text until a stopping condition.
 * 
 * @param <I> input type
 * @param <O> output type
 */
@FunctionalInterface
public interface ResponseGenerator<I, O> {
    /**
     * Generate a complete response from the given input.
     * 
     * @param model the model to use for generation
     * @param input the initial input
     * @param maxSteps maximum generation steps
     * @return generated response
     */
    O generateResponse(Object model, I input, int maxSteps);
}