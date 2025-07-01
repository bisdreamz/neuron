package dev.neuronic.net.layers;

import dev.neuronic.net.Shape;
import dev.neuronic.net.losses.Loss;
import dev.neuronic.net.optimizers.Optimizer;

import java.util.concurrent.ExecutorService;

public interface Layer {

    /**
     * Context object for caching backwards invocations.
     * Contains the inputs, preactivations, and outputs from a layer's forward pass.
     * Can be extended by specific layer types to store additional intermediate states.
     */
    public static class LayerContext {
        public final float[] inputs;         // inputs seen by the layer
        public final float[] preActivations; // preactivations from forward call  
        public final float[] outputs;        // resulting outputs from this layer
        
        public LayerContext(float[] inputs, float[] preActivations, float[] outputs) {
            this.inputs = inputs;
            this.preActivations = preActivations;
            this.outputs = outputs;
        }
        
        
        // Accessor methods for compatibility with previous record-based API
        public float[] inputs() { return inputs; }
        public float[] preActivations() { return preActivations; }
        public float[] outputs() { return outputs; }
    }

    public LayerContext forward(float[] input, boolean isTraining);

    public float[] backward(LayerContext[] stack, int stackIndex, float[] upstreamGradient);
    
    default float[] backward(LayerContext[] stack, int stackIndex, float[] upstreamGradient, 
                           Loss loss) {
        return backward(stack, stackIndex, upstreamGradient);
    }
    
    /**
     * Compute gradients without updating weights, returning both downstream gradient
     * and any parameter gradients. This enables lock-free parallel training.
     * 
     * <p>The default implementation assumes the layer has no trainable parameters
     * and simply calls backward(). Layers with parameters should override this.
     * 
     * @param stack the layer context stack from forward pass
     * @param stackIndex this layer's index in the stack
     * @param upstreamGradient gradient from the layer above
     * @param gradientConsumer callback to receive computed gradients (may be null)
     * @return gradient to pass to the layer below
     */
    default float[] computeGradient(LayerContext[] stack, int stackIndex, 
                                   float[] upstreamGradient, GradientConsumer gradientConsumer) {
        // Default: no trainable parameters, just compute downstream gradient
        return backward(stack, stackIndex, upstreamGradient);
    }
    
    /**
     * Compute gradients for output layers that include loss computation.
     * This is called when the layer is the final layer and needs to compute loss gradient.
     * 
     * @param stack the layer context stack from forward pass
     * @param stackIndex this layer's index in the stack
     * @param targets the target values for loss computation
     * @param gradientConsumer callback to receive computed gradients (may be null)
     * @return gradient to pass to the layer below
     */
    default float[] computeGradientWithTargets(LayerContext[] stack, int stackIndex,
                                              float[] targets, GradientConsumer gradientConsumer) {
        // Default implementation for output layers - compute loss gradient via backward
        return backward(stack, stackIndex, targets);
    }
    
    /**
     * Functional interface for receiving computed gradients.
     * This avoids allocating gradient objects for layers without parameters.
     */
    @FunctionalInterface
    public interface GradientConsumer {
        /**
         * Consume computed gradients for a layer.
         * 
         * @param layerIndex the layer's index in the network
         * @param weightGradients gradients for weights (null if none)
         * @param biasGradients gradients for biases (null if none)
         */
        void accept(int layerIndex, float[][] weightGradients, float[] biasGradients);
    }
    
    /**
     * Apply gradients to update this layer's parameters.
     * Only called for layers that have trainable parameters.
     * 
     * <p>The default implementation does nothing (for layers without parameters).
     * 
     * @param weightGradients gradients for weights (already scaled by 1/batchSize)
     * @param biasGradients gradients for biases (already scaled by 1/batchSize)
     */
    default void applyGradients(float[][] weightGradients, float[] biasGradients) {
        // Default: no parameters to update
    }
    
    /**
     * Get the dimensions for gradient accumulation buffers.
     * Returns null if this layer has no trainable parameters.
     * 
     * @return gradient dimensions or null
     */
    default GradientDimensions getGradientDimensions() {
        return null; // Default: no trainable parameters
    }
    
    /**
     * Simple record for gradient buffer dimensions.
     */
    public static record GradientDimensions(int weightRows, int weightCols, int biasSize) {}
    
    // Executor-aware methods with smart defaults
    default LayerContext forward(float[] input, ExecutorService executor) {
        // This default implementation assumes training mode is false for executor-based forward passes.
        // Layers that have different behavior for training/inference should override this.
        return forward(input, false);
    }
    
    default float[] backward(LayerContext[] stack, int stackIndex, float[] upstreamGradient, ExecutorService executor) {
        if (executor == null)
            return backward(stack, stackIndex, upstreamGradient);
        
        // Smart default: submit to executor and wait for result
        try {
            return executor.submit(() -> backward(stack, stackIndex, upstreamGradient)).get();
        } catch (Exception e) {
            throw new RuntimeException("Parallel backward pass failed", e);
        }
    }
    
    default float[] backward(LayerContext[] stack, int stackIndex, float[] upstreamGradient,
                             Loss loss, ExecutorService executor) {
        if (executor == null)
            return backward(stack, stackIndex, upstreamGradient, loss);
        
        // Smart default: submit to executor and wait for result
        try {
            return executor.submit(() -> backward(stack, stackIndex, upstreamGradient, loss)).get();
        } catch (Exception e) {
            throw new RuntimeException("Parallel backward pass failed", e);
        }
    }
    
    public int getOutputSize();
    
    /**
     * Get the optimizer used by this layer, if any.
     * 
     * @return the optimizer, or null if this layer doesn't use an optimizer
     */
    default Optimizer getOptimizer() {
        return null;
    }
    
    /**
     * Get the weights for this layer, if any.
     * Returns null for layers without weights.
     * @return flattened weights array or null
     */
    default float[] getWeights() {
        return null;
    }

    /**
     * Specification for creating a layer. The actual layer instance
     * is created with the correct input size during network construction.
     */
    public static interface Spec {
        Layer create(int inputSize);
        int getOutputSize();
        
        /**
         * Create a layer with the specified input size and default optimizer.
         * The default implementation ignores the default optimizer and calls create(inputSize).
         * Implementations that support optimizer inheritance should override this method.
         * 
         * @param inputSize the input size for the layer
         * @param defaultOptimizer the default optimizer from the network builder (may be null)
         * @return the created layer
         */
        default Layer create(int inputSize, Optimizer defaultOptimizer) {
            return create(inputSize);
        }
        
        /**
         * Calculate the output size given the input size.
         * This method allows sequence-aware layers to compute their actual output size.
         * 
         * <p>For most layers, output size is fixed regardless of input size.
         * For sequence layers (GRU, Embeddings), output size depends on the sequence length
         * which is derived from the input size.
         * 
         * <p>The default implementation returns the static output size for backward compatibility.
         * 
         * @param inputSize the size of the input that will be fed to this layer
         * @return the size of the output this layer will produce
         */
        default int getOutputSize(int inputSize) {
            return getOutputSize();
        }
        
        // ===== NEW SHAPE-AWARE API =====
        
        /**
         * Create a layer with the specified input shape and default optimizer.
         * 
         * <p>This shape-aware method enables proper handling of multi-dimensional data.
         * The default implementation converts to flat size for backward compatibility.
         * 
         * @param inputShape the shape of the input tensor
         * @param defaultOptimizer the default optimizer from the network builder (may be null)
         * @return the created layer
         */
        default Layer create(Shape inputShape, Optimizer defaultOptimizer) {
            // Default: flatten to 1D for backward compatibility
            return create(inputShape.toFlatSize(), defaultOptimizer);
        }
        
        /**
         * Calculate the output shape given the input shape.
         * 
         * <p>This method enables proper shape inference for multi-dimensional layers.
         * The default implementation assumes 1D output for backward compatibility.
         * 
         * @param inputShape the shape of the input tensor
         * @return the shape of the output tensor
         */
        default Shape getOutputShape(Shape inputShape) {
            // Default: assume 1D output using the size calculation
            return Shape.vector(getOutputSize(inputShape.toFlatSize()));
        }
        
        /**
         * Validate that the input shape is acceptable for this layer.
         * 
         * <p>Override this method to add shape constraints (e.g., Conv2D requires 3D input).
         * The default implementation accepts any shape.
         * 
         * @param inputShape the shape to validate
         * @throws IllegalArgumentException if the shape is not valid for this layer
         */
        default void validateInputShape(Shape inputShape) {
            // Default: accept any shape
        }
        
        /**
         * Check if this layer spec prefers shape-based creation.
         * 
         * <p>Layers that override this to return true indicate they have proper
         * shape handling and should use the shape-aware API path.
         * 
         * @return true if this layer has native shape support
         */
        default boolean prefersShapeAPI() {
            return false;
        }
    }

}
