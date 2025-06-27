package dev.neuronic.net.simple;

import dev.neuronic.net.NeuralNet;
import dev.neuronic.net.layers.Layer;
import dev.neuronic.net.outputs.LinearRegressionOutput;
import dev.neuronic.net.outputs.SigmoidBinaryCrossEntropyOutput;
import dev.neuronic.net.outputs.SoftmaxCrossEntropyOutput;
import dev.neuronic.net.serialization.Serializable;
import java.util.*;

/**
 * Base class and factory for creating type-safe neural network wrappers with automatic feature handling.
 * 
 * <p><b>What it does:</b> Provides factory methods to create specialized neural network wrappers 
 * that eliminate casting, ensure type safety, and provide clean APIs for different use cases.
 * 
 * <p><b>Why use SimpleNet factories:</b>
 * <ul>
 *   <li><b>Type safety:</b> No casting needed - each wrapper returns the correct type</li>
 *   <li><b>Performance:</b> Specialized wrappers use primitives (int, float) instead of boxed types</li>
 *   <li><b>Clear intent:</b> Factory method names make the use case obvious</li>
 *   <li><b>Error prevention:</b> Validates neural network configuration at creation time</li>
 *   <li><b>Automatic features:</b> Handles dictionary building, feature mapping, output conversion</li>
 * </ul>
 * 
 * <p><b>Available Factory Methods:</b>
 * <ul>
 *   <li><b>{@link #ofIntClassification(NeuralNet)}:</b> For MNIST-style integer classification</li>
 *   <li><b>{@link #ofStringClassification(NeuralNet)}:</b> For text classification with string labels</li>
 *   <li><b>{@link #ofFloatRegression(NeuralNet)}:</b> For single-value regression (price prediction, etc.)</li>
 *   <li><b>{@link #ofMultiFloatRegression(NeuralNet, String[])}:</b> For multi-output regression with named outputs</li>
 *   <li><b>{@link #ofMultiFloatRegression(NeuralNet)}:</b> For multi-output regression without named outputs</li>
 * </ul>
 * 
 * <p><b>Example - MNIST Integer Classification:</b>
 * <pre>{@code
 * // Neural network for MNIST digit recognition
 * NeuralNet net = NeuralNet.newBuilder()
 *     .input(784)  // 28x28 MNIST images
 *     .layer(Layers.hiddenDenseRelu(256))
 *     .layer(Layers.hiddenDenseRelu(64))
 *     .output(Layers.outputSoftmaxCrossEntropy(10));  // 10 digit classes
 * 
 * // Type-safe classifier - returns primitive int (no casting!)
 * SimpleNetInt classifier = SimpleNet.ofIntClassification(net);
 * 
 * // Train with MNIST data - automatic label discovery
 * classifier.train(mnistPixels1, 7);  // First time seeing digit 7
 * classifier.train(mnistPixels2, 3);  // First time seeing digit 3
 * 
 * // Predict - returns primitive int directly
 * int predictedDigit = classifier.predict(testPixels);  // No casting needed!
 * }</pre>
 * 
 * <p><b>Example - Text Classification with String Labels:</b>
 * <pre>{@code
 * // Email sentiment classifier with mixed features
 * NeuralNet net = NeuralNet.newBuilder()
 *     .input(3)
 *     .layer(Layers.inputMixed(optimizer,
 *         Feature.embedding(10000, 128),  // email_text → embedding
 *         Feature.oneHot(4),              // sender_type → one-hot
 *         Feature.passthrough()           // length → passthrough
 *     ))
 *     .layer(Layers.hiddenDenseRelu(256))
 *     .output(Layers.outputSoftmaxCrossEntropy(3));
 * 
 * // Type-safe classifier - returns String (no casting!)
 * SimpleNetString classifier = SimpleNet.ofStringClassification(net);
 * 
 * // Train with string labels - automatic discovery
 * classifier.train(Map.of("email_text", "Great product!", "sender_type", 1, "length", 127.0f), "positive");
 * classifier.train(Map.of("email_text", "Terrible service", "sender_type", 2, "length", 89.0f), "negative");
 * 
 * // Predict - returns String directly
 * String sentiment = classifier.predict(emailFeatures);  // "positive", "negative", or "neutral"
 * }</pre>
 * 
 * <p><b>Example - Single Regression:</b>
 * <pre>{@code
 * // House price prediction
 * NeuralNet net = NeuralNet.newBuilder()
 *     .input(4)
 *     .layer(Layers.inputMixed(optimizer, ...))
 *     .output(Layers.outputLinearRegression(1));  // Single price output
 * 
 * // Type-safe regressor - returns primitive float (no casting!)
 * SimpleNetFloat pricer = SimpleNet.ofFloatRegression(net);
 * 
 * float price = pricer.predict(houseFeatures);  // No casting needed!
 * }</pre>
 * 
 * <p><b>Benefits of Factory Approach:</b>
 * <ul>
 *   <li><b>No casting:</b> Each wrapper returns the exact type you need</li>
 *   <li><b>Validation:</b> Factory methods validate neural network configuration</li>
 *   <li><b>Performance:</b> Uses primitives (int, float) instead of boxed types</li>
 *   <li><b>Automatic dictionaries:</b> Builds label/feature mappings during training</li>
 *   <li><b>Clear intent:</b> Method names make the use case obvious</li>
 * </ul>
 */
public abstract class SimpleNet implements Serializable {
    
    // Instance fields for shared functionality
    protected final NeuralNet underlyingNet;
    protected final Set<String> outputNames;
    protected final Map<String, Integer> outputNameToIndex;
    
    /**
     * Protected constructor for subclasses.
     * 
     * @param underlyingNet the neural network
     * @param outputNames optional set of output names (can be null)
     */
    protected SimpleNet(NeuralNet underlyingNet, Set<String> outputNames) {
        this.underlyingNet = underlyingNet;
        
        if (outputNames != null && !outputNames.isEmpty()) {
            this.outputNames = new LinkedHashSet<>(outputNames); // Preserve order
            this.outputNameToIndex = new HashMap<>();
            int index = 0;
            for (String name : this.outputNames) {
                outputNameToIndex.put(name, index++);
            }
            
            // Validate output count matches
            if (outputNames.size() != underlyingNet.getOutputLayer().getOutputSize()) {
                throw new IllegalArgumentException(String.format(
                    "Output names length (%d) must match network output size (%d)",
                    outputNames.size(), underlyingNet.getOutputLayer().getOutputSize()));
            }
        } else {
            this.outputNames = null;
            this.outputNameToIndex = null;
        }
    }
    
    
    // ===============================
    // SHARED METHODS FOR NAMED OUTPUTS
    // ===============================
    
    /**
     * Train with named targets using Map.
     * Requires output names to be configured during construction.
     * 
     * @param input input data (type depends on network configuration)
     * @param namedTargets map of output name to target value
     * @throws IllegalStateException if output names not configured
     * @throws IllegalArgumentException if map doesn't match configured outputs
     */
    public void train(Object input, Map<String, Float> namedTargets) {
        if (outputNames == null) {
            throw new IllegalStateException(
                "Cannot use named targets without output names. " +
                "Provide output names during construction.");
        }
        
        float[] targets = convertNamedTargets(namedTargets);
        trainInternal(input, targets);
    }
    
    /**
     * Predict and return results as a named map.
     * Requires output names to be configured during construction.
     * 
     * @param input input data (type depends on network configuration)
     * @return map of output name to predicted value
     * @throws IllegalStateException if output names not configured
     */
    public Map<String, Float> predictNamed(Object input) {
        if (outputNames == null) {
            throw new IllegalStateException(
                "Cannot use predictNamed() without output names. " +
                "Provide output names during construction.");
        }
        
        float[] predictions = predictInternal(input);
        Map<String, Float> namedResults = new LinkedHashMap<>();
        
        int index = 0;
        for (String name : outputNames) {
            namedResults.put(name, predictions[index++]);
        }
        
        return namedResults;
    }
    
    /**
     * Convert named targets to array in correct order.
     */
    protected float[] convertNamedTargets(Map<String, Float> namedTargets) {
        if (namedTargets.size() != outputNames.size()) {
            throw new IllegalArgumentException(String.format(
                "Target map has %d entries but %d outputs configured. Expected: %s",
                namedTargets.size(), outputNames.size(), outputNames));
        }
        
        float[] targets = new float[outputNames.size()];
        
        for (Map.Entry<String, Float> entry : namedTargets.entrySet()) {
            String name = entry.getKey();
            Float value = entry.getValue();
            
            Integer index = outputNameToIndex.get(name);
            if (index == null) {
                throw new IllegalArgumentException(
                    "Unknown output name: '" + name + "'. Valid outputs: " + outputNames);
            }
            
            if (value == null)
                throw new IllegalArgumentException("Target value for '" + name + "' cannot be null");
            
            targets[index] = value;
        }
        
        return targets;
    }
    
    /**
     * Get the underlying neural network.
     */
    public NeuralNet getNetwork() {
        return underlyingNet;
    }
    
    /**
     * Check if output names are configured.
     */
    public boolean hasOutputNames() {
        return outputNames != null;
    }
    
    /**
     * Get configured output names (defensive copy).
     */
    public Set<String> getOutputNames() {
        return outputNames != null ? new LinkedHashSet<>(outputNames) : null;
    }
    
    // Abstract methods that subclasses must implement
    protected abstract void trainInternal(Object input, float[] targets);
    protected abstract float[] predictInternal(Object input);
    
    // ===============================
    // FACTORY METHODS
    // ===============================
    
    /**
     * Create a type-safe classifier for integer labels (MNIST, CIFAR-10, etc.).
     * 
     * <p><b>Perfect for:</b> Datasets where class labels are integers (0, 1, 2, ...).
     * Returns primitive int - no casting or boxing overhead.
     * 
     * <p><b>Example - MNIST:</b>
     * <pre>{@code
     * NeuralNet net = NeuralNet.newBuilder()
     *     .input(784)
     *     .layer(Layers.hiddenDenseRelu(256))
     *     .output(Layers.outputSoftmaxCrossEntropy(10));
     * 
     * SimpleNetInt classifier = SimpleNet.ofIntClassification(net);
     * 
     * // Train and predict with integers
     * classifier.train(pixels, 7);
     * int digit = classifier.predict(testPixels);  // Returns 0-9
     * }</pre>
     * 
     * <p><b>Validation:</b> Ensures neural network has classification output layer.
     * 
     * @param net neural network configured for classification
     * @return type-safe classifier that returns primitive int
     * @throws IllegalArgumentException if network is not suitable for classification
     */
    public static SimpleNetInt ofIntClassification(NeuralNet net) {
        validateClassificationNetwork(net);
        return new SimpleNetInt(net, null);
    }
    
    /**
     * Create a type-safe classifier for integer labels with optional output name.
     * 
     * @param net neural network configured for classification
     * @param outputName optional name for the output (e.g., "digit", "class")
     * @return type-safe classifier that returns primitive int
     * @throws IllegalArgumentException if network is not suitable for classification
     */
    public static SimpleNetInt ofIntClassification(NeuralNet net, String outputName) {
        validateClassificationNetwork(net);
        Set<String> names = outputName != null ? Set.of(outputName) : null;
        return new SimpleNetInt(net, names);
    }
    
    /**
     * Create a type-safe classifier for string labels (sentiment, text classification, etc.).
     * 
     * <p><b>Perfect for:</b> Text classification, sentiment analysis, named entity recognition.
     * Returns String labels - no casting needed.
     * 
     * <p><b>Example - Sentiment Analysis:</b>
     * <pre>{@code
     * SimpleNetString classifier = SimpleNet.ofStringClassification(net);
     * 
     * // Train with string labels
     * classifier.train(textFeatures, "positive");
     * classifier.train(textFeatures2, "negative");
     * 
     * String sentiment = classifier.predict(newText);  // "positive", "negative", "neutral"
     * }</pre>
     * 
     * @param net neural network configured for classification
     * @return type-safe classifier that returns String labels
     * @throws IllegalArgumentException if network is not suitable for classification
     */
    public static SimpleNetString ofStringClassification(NeuralNet net) {
        validateClassificationNetwork(net);
        return new SimpleNetString(net, null);
    }
    
    /**
     * Create a type-safe classifier for string labels with optional output name.
     * 
     * @param net neural network configured for classification
     * @param outputName optional name for the output (e.g., "sentiment", "category")
     * @return type-safe classifier that returns String labels
     * @throws IllegalArgumentException if network is not suitable for classification
     */
    public static SimpleNetString ofStringClassification(NeuralNet net, String outputName) {
        validateClassificationNetwork(net);
        Set<String> names = outputName != null ? Set.of(outputName) : null;
        return new SimpleNetString(net, names);
    }
    
    /**
     * Create a type-safe regressor for single-value prediction (price, score, rating, etc.).
     * 
     * <p><b>Perfect for:</b> Predicting single numerical values like prices, scores, temperatures.
     * Returns primitive float - no casting or boxing overhead.
     * 
     * <p><b>Example - House Price Prediction:</b>
     * <pre>{@code
     * NeuralNet net = NeuralNet.newBuilder()
     *     .input(4)
     *     .layer(Layers.inputMixed(optimizer, ...))
     *     .output(Layers.outputLinearRegression(1));  // Single output
     * 
     * SimpleNetFloat pricer = SimpleNet.ofFloatRegression(net);
     * 
     * float price = pricer.predict(houseFeatures);  // Returns primitive float
     * }</pre>
     * 
     * @param net neural network with single regression output
     * @return type-safe regressor that returns primitive float
     * @throws IllegalArgumentException if network doesn't have single regression output
     */
    public static SimpleNetFloat ofFloatRegression(NeuralNet net) {
        validateSingleRegressionNetwork(net);
        return new SimpleNetFloat(net, null);
    }
    
    /**
     * Create a type-safe regressor for single-value prediction with optional output name.
     * 
     * @param net neural network with single regression output
     * @param outputName optional name for the output (e.g., "price", "score")
     * @return type-safe regressor that returns primitive float
     * @throws IllegalArgumentException if network doesn't have single regression output
     */
    public static SimpleNetFloat ofFloatRegression(NeuralNet net, String outputName) {
        validateSingleRegressionNetwork(net);
        Set<String> names = outputName != null ? Set.of(outputName) : null;
        return new SimpleNetFloat(net, names);
    }
    
    /**
     * Create a type-safe language model for next-word prediction and text generation.
     * 
     * <p><b>Perfect for:</b> Language modeling, text generation, autocomplete.
     * Input and output share the same vocabulary.
     * 
     * <p><b>Example - Text Generation:</b>
     * <pre>{@code
     * NeuralNet net = NeuralNet.newBuilder()
     *     .input(30)  // sequence length
     *     .layer(Layers.inputSequenceEmbedding(30, 10000, 128))
     *     .layer(Layers.hiddenGruLast(256))
     *     .output(Layers.outputSoftmaxCrossEntropy(10000));
     * 
     * SimpleNetLanguageModel lm = SimpleNet.ofLanguageModel(net);
     * 
     * lm.train(new String[]{"the", "cat", "sat", "on", "the"}, "mat");
     * String next = lm.predictNext(new String[]{"the", "dog", "sat", "on", "the"});
     * }</pre>
     * 
     * @param net neural network with InputSequenceEmbeddingLayer
     * @return type-safe language model
     * @throws IllegalArgumentException if network doesn't have InputSequenceEmbeddingLayer
     */
    public static SimpleNetLanguageModel ofLanguageModel(NeuralNet net) {
        return new SimpleNetLanguageModel(net, null);
    }
    
    /**
     * Create a type-safe language model with optional output name.
     * 
     * @param net neural network with InputSequenceEmbeddingLayer
     * @param outputName optional name for the output (e.g., "next_word")
     * @return type-safe language model
     * @throws IllegalArgumentException if network doesn't have InputSequenceEmbeddingLayer
     */
    public static SimpleNetLanguageModel ofLanguageModel(NeuralNet net, String outputName) {
        Set<String> names = outputName != null ? Set.of(outputName) : null;
        return new SimpleNetLanguageModel(net, names);
    }
    
    /**
     * Create a type-safe regressor for multi-output prediction (coordinates, portfolios, multi-metrics).
     * 
     * <p><b>Perfect for:</b> Predicting multiple numerical values simultaneously.
     * Returns float[] - no casting needed.
     * 
     * <p><b>Example - Portfolio Allocation:</b>
     * <pre>{@code
     * NeuralNet net = NeuralNet.newBuilder()
     *     .input(4)
     *     .layer(Layers.inputMixed(optimizer, ...))
     *     .output(Layers.outputLinearRegression(4));  // 4 asset allocations
     * 
     * String[] assetNames = {"stocks", "bonds", "real_estate", "cash"};
     * SimpleNetMultiFloat allocator = SimpleNet.ofMultiFloatRegression(net, assetNames);
     * 
     * float[] allocation = allocator.predict(investorProfile);  // Returns float[]
     * Map<String, Float> namedAllocation = allocator.predictNamed(investorProfile);  // Named results
     * }</pre>
     * 
     * @param net neural network with multi-output regression
     * @param outputNames optional names for each output (null for unnamed outputs)
     * @return type-safe regressor that returns float[]
     * @throws IllegalArgumentException if network doesn't have regression output or names don't match output count
     */
    public static SimpleNetMultiFloat ofMultiFloatRegression(NeuralNet net, String[] outputNames) {
        validateMultiRegressionNetwork(net);
        Set<String> names = null;
        if (outputNames != null && outputNames.length > 0) {
            names = new LinkedHashSet<>(Arrays.asList(outputNames));
            if (names.size() != outputNames.length) {
                throw new IllegalArgumentException("Output names must be unique");
            }
        }
        return new SimpleNetMultiFloat(net, names);
    }
    
    /**
     * Create a type-safe regressor for multi-output prediction without named outputs.
     * 
     * @param net neural network with multi-output regression
     * @return type-safe regressor that returns float[]
     * @throws IllegalArgumentException if network doesn't have regression output
     */
    public static SimpleNetMultiFloat ofMultiFloatRegression(NeuralNet net) {
        validateMultiRegressionNetwork(net);
        return new SimpleNetMultiFloat(net);
    }
    
    /**
     * Create a type-safe regressor for multi-output prediction with named outputs using Set.
     * 
     * @param net neural network with multi-output regression
     * @param outputNames set of unique output names
     * @return type-safe regressor that returns float[]
     * @throws IllegalArgumentException if network doesn't have regression output or names don't match output count
     */
    public static SimpleNetMultiFloat ofMultiFloatRegression(NeuralNet net, Set<String> outputNames) {
        validateMultiRegressionNetwork(net);
        return new SimpleNetMultiFloat(net, outputNames);
    }
    
    // ===============================
    // VALIDATION HELPERS
    // ===============================
    
    private static void validateClassificationNetwork(NeuralNet net) {
        Layer outputLayer = net.getOutputLayer();
        
        if (!(outputLayer instanceof SoftmaxCrossEntropyOutput) &&
            !(outputLayer instanceof SigmoidBinaryCrossEntropyOutput)) {
            throw new IllegalArgumentException(
                "For classification, use SoftmaxCrossEntropy or SigmoidBinary output layer. " +
                "Found: " + outputLayer.getClass().getSimpleName());
        }
        
        // Note: SimpleNet handles output formatting itself using the new prediction methods
    }
    
    private static void validateSingleRegressionNetwork(NeuralNet net) {
        Layer outputLayer = net.getOutputLayer();
        
        if (!(outputLayer instanceof LinearRegressionOutput)) {
            throw new IllegalArgumentException(
                "For regression, use LinearRegression output layer. " +
                "Found: " + outputLayer.getClass().getSimpleName());
        }
        
        if (outputLayer.getOutputSize() != 1) {
            throw new IllegalArgumentException(
                "For single regression, use outputLinearRegression(1). " +
                "Found output size: " + outputLayer.getOutputSize() + 
                ". Use ofMultiFloatRegression() for multiple outputs.");
        }
    }
    
    private static void validateMultiRegressionNetwork(NeuralNet net) {
        Layer outputLayer = net.getOutputLayer();
        
        if (!(outputLayer instanceof LinearRegressionOutput)) {
            throw new IllegalArgumentException(
                "For regression, use LinearRegression output layer. " +
                "Found: " + outputLayer.getClass().getSimpleName());
        }
        
        if (outputLayer.getOutputSize() < 1) {
            throw new IllegalArgumentException(
                "For multi-output regression, use outputLinearRegression(n) where n > 0. " +
                "Found output size: " + outputLayer.getOutputSize());
        }
    }
}