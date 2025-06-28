package dev.neuronic.net.outputs;

/**
 * Marker interface for regression output layers.
 * 
 * <p>This interface identifies layers that perform regression tasks,
 * allowing SimpleNet and other components to validate network configurations
 * without hard-coding specific implementation classes.
 * 
 * <p>All regression output layers (LinearRegressionOutput, HuberRegressionOutput, etc.)
 * should implement this interface to be recognized as valid regression outputs.
 */
public interface RegressionOutput {
    // Marker interface - no methods required
}