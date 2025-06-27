package dev.neuronic.net.repl;

/**
 * Formats model outputs for display to the user.
 * 
 * @param <T> the type of output the model produces
 */
@FunctionalInterface
public interface OutputFormatter<T> {
    /**
     * Format model output for display.
     * 
     * @param output raw model output
     * @return formatted string for display
     */
    String format(T output);
}