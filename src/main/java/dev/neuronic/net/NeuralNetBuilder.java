package dev.neuronic.net;

import dev.neuronic.net.layers.Layer;
import dev.neuronic.net.optimizers.Optimizer;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.ExecutorService;

/**
 * Builder for creating neural networks with a clean, type-safe API.
 * 
 * Encourages proper separation between hidden layers and output layers.
 * Output layers handle their own loss computation, preventing configuration errors.
 * 
 * Example usage:
 * <pre>{@code
 * // Standard network returning raw probabilities
 * NeuralNet net = NeuralNet.newBuilder()
 *     .input(784)
 *     .layer(Layers.hiddenDenseRelu(256, optimizer))
 *     .layer(Layers.hiddenDenseRelu(128, optimizer))
 *     .output(Layers.outputSoftmaxCrossEntropy(10, optimizer));
 * float[] probabilities = net.predict(input); // Returns [0.1, 0.8, 0.1, ...]
 * 
 * // Classification network returning class indices directly
 * NeuralNet classifier = NeuralNet.newBuilder()
 *     .input(784)
 *     .layer(Layers.hiddenDenseRelu(256, optimizer))
 *     .layer(Layers.hiddenDenseRelu(128, optimizer))
 *     .argMax(1)  // Enable argmax mode
 *     .output(Layers.outputSoftmaxCrossEntropy(10, optimizer));
 * float[] result = classifier.predict(input); // Returns [1.0] (class index)
 * int predictedClass = (int) result[0];
 * 
 * // With internal parallelism for large models
 * NeuralNet bigNet = NeuralNet.newBuilder()
 *     .input(784)
 *     .layer(Layers.hiddenDenseRelu(10000, optimizer))
 *     .executor(myExecutor)
 *     .argMax(1)
 *     .output(Layers.outputSoftmaxCrossEntropy(10, optimizer));
 * }</pre>
 */
public class NeuralNetBuilder {

    private List<Layer.Spec> layerSpecs = new ArrayList<>();
    private int inputSize;
    private Shape inputShape; // NEW: track shape alongside size
    private ExecutorService executor = null;
    private Optimizer defaultOptimizer = null;
    private float globalGradientClipNorm = 10.0f; // Safety default to prevent NaN

    public NeuralNetBuilder input(int size) {
        this.inputSize = size;
        this.inputShape = Shape.vector(size); // Store as 1D shape
        return this;
    }
    
    /**
     * Sets the input shape for the network.
     * 
     * <p>This shape-aware method enables proper handling of multi-dimensional inputs
     * like sequences, images, or higher-dimensional tensors.
     * 
     * <p><b>Examples:</b>
     * <pre>{@code
     * // Sequence input: 50 timesteps, each with a single feature (token ID)
     * .input(Shape.sequence(50, 1))
     * 
     * // Image input: 28x28 grayscale image
     * .input(Shape.image(28, 28, 1))
     * 
     * // Raw vector (equivalent to .input(784))
     * .input(Shape.vector(784))
     * }</pre>
     * 
     * @param shape the shape of the input tensor
     * @return this builder for method chaining
     */
    public NeuralNetBuilder input(Shape shape) {
        this.inputShape = shape;
        this.inputSize = shape.toFlatSize(); // Maintain compatibility
        return this;
    }
    
    /**
     * Sets the default optimizer for all layers in this network.
     * 
     * <p><b>What is this?</b> An optimizer controls how your neural network learns from its mistakes.
     * Setting a default optimizer means all layers will use the same learning algorithm unless
     * you explicitly override it for specific layers.
     * 
     * <p><b>Why set a default?</b> Most neural networks use the same optimizer for all layers.
     * Setting a default saves you from specifying the optimizer for every single layer.
     * 
     * <p><b>Common optimizers to use:</b>
     * <ul>
     *   <li><b>AdamWOptimizer</b> - Best default choice. Modern Adam with better weight decay for superior generalization.</li>
     *   <li><b>AdamOptimizer</b> - Original Adam optimizer. Still good but AdamW is generally better.</li>
     *   <li><b>SgdOptimizer</b> - Simple and fast. Good when you have lots of training data.</li>
     * </ul>
     * 
     * <p><b>Example:</b>
     * <pre>{@code
     * // All layers will use AdamW optimizer with learning rate 0.001 and weight decay 0.01
     * NeuralNet net = NeuralNet.newBuilder()
     *     .setDefaultOptimizer(new AdamWOptimizer(0.001f, 0.01f))
     *     .input(784)
     *     .layer(Layers.hiddenDenseRelu(256))     // Uses AdamW(0.001f, 0.01f)
     *     .layer(Layers.hiddenDenseRelu(128))     // Uses AdamW(0.001f, 0.01f)
     *     .output(Layers.outputSoftmaxCrossEntropy(10)); // Uses AdamW(0.001f, 0.01f)
     * }</pre>
     * 
     * <p><b>Individual layer override:</b> You can still set different optimizers for specific layers:
     * <pre>{@code
     * .layer(Layers.hiddenDenseRelu(256).optimizer(new AdamOptimizer(0.001f))) // Legacy Adam for this layer
     * }</pre>
     * 
     * @param optimizer the optimizer to use for all layers by default
     * @return this builder for method chaining
     */
    public NeuralNetBuilder setDefaultOptimizer(Optimizer optimizer) {
        this.defaultOptimizer = optimizer;
        return this;
    }
    
    /**
     * Set the global gradient clipping norm.
     * 
     * <p>Gradient clipping prevents gradient explosion by limiting the L2 norm of all gradients.
     * This is essential for training RNNs, GRUs, and can help stabilize any deep network.
     * 
     * <p><b>Default:</b> 10.0f (safety rail against NaN)
     * 
     * <p><b>Recommended values:</b>
     * <ul>
     *   <li>RNNs/GRUs/LSTMs: 1.0 - 5.0</li>
     *   <li>Transformers: 1.0 - 2.0</li>
     *   <li>CNNs: 5.0 - 10.0 (if needed)</li>
     *   <li>Feedforward: Usually not needed, but 10.0 prevents NaN</li>
     *   <li>Disable: 0.0</li>
     * </ul>
     * 
     * <p><b>Example:</b>
     * <pre>{@code
     * NeuralNet rnn = NeuralNet.newBuilder()
     *     .input(100)
     *     .withGlobalGradientClipping(1.0f)  // Essential for RNNs
     *     .layer(Layers.hiddenGruLast(64))
     *     .output(Layers.outputSoftmaxCrossEntropy(10));
     * }</pre>
     * 
     * @param maxNorm maximum allowed L2 norm for gradients (0 to disable)
     * @return this builder
     */
    public NeuralNetBuilder withGlobalGradientClipping(float maxNorm) {
        if (maxNorm < 0) {
            throw new IllegalArgumentException("Gradient clip norm must be >= 0, got: " + maxNorm);
        }
        this.globalGradientClipNorm = maxNorm;
        return this;
    }

    /**
     * Add a hidden layer to the network.
     * Use Layers.hiddenXxx() methods for type-safe layer creation.
     */
    public NeuralNetBuilder layer(Layer.Spec layerSpec) {
        layerSpecs.add(layerSpec);
        return this;
    }
    
    /**
     * Set the executor service for optional internal parallelism.
     * If not set, all operations run on the calling thread.
     * 
     * @param executor executor service for parallel layer operations
     * @return this builder for method chaining
     */
    public NeuralNetBuilder executor(ExecutorService executor) {
        this.executor = executor;
        return this;
    }
    

    /**
     * Set the output layer and build the network.
     * Use Layers.outputXxx() methods for proper output layer creation.
     */
    public NeuralNet output(Layer.Spec outputLayerSpec) {
        // Add output layer to the list
        layerSpecs.add(outputLayerSpec);
        
        // Build all layers
        Layer[] layers = new Layer[layerSpecs.size()];
        int currentInputSize = inputSize;
        Shape currentShape = inputShape;

        for (int i = 0; i < layerSpecs.size(); i++) {
            Layer.Spec spec = layerSpecs.get(i);
            
            // Use shape-aware API if available and preferred
            if (currentShape != null && spec.prefersShapeAPI()) {
                // Validate shape first
                spec.validateInputShape(currentShape);
                
                // Create with shape information
                layers[i] = spec.create(currentShape, defaultOptimizer);
                
                // Get output shape for next layer
                currentShape = spec.getOutputShape(currentShape);
                currentInputSize = currentShape.toFlatSize();
            } else {
                // Fall back to legacy API
                layers[i] = spec.create(currentInputSize, defaultOptimizer);
                currentInputSize = spec.getOutputSize(currentInputSize);
                
                // Try to maintain shape information
                if (currentShape != null) {
                    currentShape = Shape.vector(currentInputSize);
                }
            }
        }

        return new NeuralNet(layers, executor, globalGradientClipNorm);
    }
    

    /**
     * @deprecated Use output() method instead. Output layers should handle their own loss computation.
     */
    @Deprecated
    public NeuralNet build() {
        throw new UnsupportedOperationException(
            "Use output() method instead. Example: .output(Layers.outputSoftmaxCrossEntropy(10, optimizer))");
    }

}