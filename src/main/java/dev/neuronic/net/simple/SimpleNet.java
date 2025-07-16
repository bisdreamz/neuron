package dev.neuronic.net.simple;

import dev.neuronic.net.NeuralNet;
import dev.neuronic.net.layers.Layer;
import dev.neuronic.net.outputs.LinearRegressionOutput;
import dev.neuronic.net.outputs.SigmoidBinaryCrossEntropyOutput;
import dev.neuronic.net.outputs.SoftmaxCrossEntropyOutput;
import dev.neuronic.net.outputs.RegressionOutput;
import dev.neuronic.net.serialization.Serializable;
import dev.neuronic.net.losses.Loss;
import dev.neuronic.net.common.Utils;
import dev.neuronic.net.training.BatchTrainer;
import dev.neuronic.net.training.TrainingCallback;
import dev.neuronic.net.training.TrainingMetrics;
import dev.neuronic.net.training.EarlyStoppingCallback;
import dev.neuronic.net.training.ModelCheckpointCallback;
import dev.neuronic.net.training.VisualizationCallback;
import dev.neuronic.net.layers.MixedFeatureInputLayer;
import dev.neuronic.net.layers.Feature;
import dev.neuronic.net.Dictionary;
import dev.neuronic.net.LRUDictionary;
import java.util.*;
import java.util.concurrent.ConcurrentHashMap;

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
 * 
 * @param <T> the type of the target/label for this SimpleNet variant
 */
public abstract class SimpleNet<T> implements Serializable {
    
    // Instance fields for shared functionality
    protected final NeuralNet underlyingNet;
    protected final Set<String> outputNames;
    protected final Map<String, Integer> outputNameToIndex;
    
    // Feature mapping fields (common to all subclasses)
    protected final boolean usesFeatureMapping;
    protected final String[] featureNames;
    protected final ConcurrentHashMap<String, Dictionary> featureDictionaries;
    protected final Feature[] features;
    
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
        
        // Initialize feature mapping fields
        Layer inputLayer = underlyingNet.getInputLayer();
        if (inputLayer instanceof MixedFeatureInputLayer) {
            this.usesFeatureMapping = true;
            MixedFeatureInputLayer mixedLayer = (MixedFeatureInputLayer) inputLayer;
            this.features = mixedLayer.getFeatures();
            
            // Get feature names and replace nulls with generated names
            String[] originalNames = mixedLayer.getFeatureNames();
            this.featureNames = new String[originalNames.length];
            for (int i = 0; i < originalNames.length; i++) {
                this.featureNames[i] = (originalNames[i] != null) ? originalNames[i] : "feature_" + i;
            }
            
            this.featureDictionaries = new ConcurrentHashMap<>();
            
            // Initialize dictionaries for features that need them
            for (int i = 0; i < featureNames.length; i++) {
                if (features[i].getType() == Feature.Type.EMBEDDING || 
                    features[i].getType() == Feature.Type.ONEHOT) {
                    // Use LRUDictionary if feature requests it
                    Dictionary dict = features[i].isLRU() 
                        ? new LRUDictionary(features[i].getMaxUniqueValues(), features[i].getMaxUniqueValues())
                        : new Dictionary(features[i].getMaxUniqueValues());
                    featureDictionaries.put(featureNames[i], dict);
                }
            }
        } else {
            this.usesFeatureMapping = false;
            this.features = null;
            this.featureNames = null;
            this.featureDictionaries = null;
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
    
    // ===============================
    // TYPE-SAFE PREDICTION METHODS
    // ===============================
    
    /**
     * Predict using a float array input.
     * Works for both simple models and mixed feature models.
     * 
     * @param input raw float array
     * @return prediction result (type depends on subclass)
     */
    public Object predict(float[] input) {
        return predictFromArray(input);
    }
    
    /**
     * Predict using a Map input.
     * For models with mixed features that use named inputs.
     * 
     * @param input map of feature names to values
     * @return prediction result (type depends on subclass)
     */
    public Object predict(Map<String, Object> input) {
        if (!usesFeatureMapping) {
            throw new IllegalArgumentException(
                "This model does not use mixed features. Use predict(float[]) instead.");
        }
        return predictFromMap(input);
    }
    
    // Subclasses override these to provide type-safe returns
    protected abstract Object predictFromArray(float[] input);
    protected abstract Object predictFromMap(Map<String, Object> input);
    
    /**
     * Helper method to convert input and call the appropriate predict method.
     * Used internally by subclasses.
     */
    protected float[] convertAndGetModelInput(Object input) {
        if (input instanceof float[]) {
            return convertFromFloatArray((float[]) input);
        } else if (input instanceof Map) {
            @SuppressWarnings("unchecked")
            Map<String, Object> mapInput = (Map<String, Object>) input;
            return convertFromMap(mapInput);
        } else {
            throw new IllegalArgumentException(
                "Input must be float[] or Map<String, Object>. Got: " + 
                (input == null ? "null" : input.getClass().getSimpleName()));
        }
    }
    
    // ===============================
    // COMMON INPUT CONVERSION
    // ===============================
    
    /**
     * Convert float array input to model input.
     * For mixed features, uses dictionaries to map float values to indices.
     * 
     * @param input raw float array
     * @return converted float array ready for the model
     */
    protected float[] convertFromFloatArray(float[] input) {
        if (!usesFeatureMapping) {
            // Simple model - just return as-is
            return input;
        }
        
        // Mixed features - apply dictionary mapping
        if (input.length != features.length) {
            throw new IllegalArgumentException(String.format(
                "Input array has %d elements but %d features were configured. " +
                "Expected input format: [feature0_value, feature1_value, ..., feature%d_value]",
                input.length, features.length, features.length - 1));
        }
        
        float[] modelInput = new float[features.length];
        for (int i = 0; i < features.length; i++) {
            Feature feature = features[i];
            switch (feature.getType()) {
                case EMBEDDING:
                case ONEHOT:
                    // Use the float value as dictionary key
                    Dictionary dict = featureDictionaries.get(featureNames[i]);
                    try {
                        int index = dict.getIndex(input[i]); // Float auto-boxed to Float object
                        modelInput[i] = (float) index;
                    } catch (IllegalStateException e) {
                        // Dictionary is full - provide helpful error message
                        if (!features[i].isLRU() && e.getMessage().contains("Dictionary is full")) {
                            throw new IllegalStateException(String.format(
                                "Dictionary for feature '%s' has grown to %d entries, exceeding maxUniqueValues=%d. " +
                                "This indicates unbounded vocabulary growth. Consider: " +
                                "1) Using Feature.hashedEmbedding() instead for high-cardinality features, " +
                                "2) Pre-processing data to limit unique values, or " +
                                "3) Using Feature.embeddingLRU() or Feature.oneHotLRU() for online learning.",
                                featureNames[i], dict.size(), features[i].getMaxUniqueValues()));
                        }
                        throw e; // Re-throw if it's a different error
                    }
                    break;
                default:
                    // PASSTHROUGH, AUTO_NORMALIZE, SCALE_BOUNDED
                    modelInput[i] = input[i];
                    break;
            }
        }
        return modelInput;
    }
    
    /**
     * Convert map input to model input.
     * Uses feature names to look up values and apply dictionaries.
     * 
     * @param input map of feature names to values
     * @return converted float array ready for the model
     */
    protected float[] convertFromMap(Map<String, Object> input) {
        if (!usesFeatureMapping) {
            throw new IllegalArgumentException(
                "Map input requires mixed features model. Use float[] input instead.");
        }
        
        // Check if Map input is allowed (subclasses can override)
        validateMapInput();
        
        if (input.size() != featureNames.length) {
            throw new IllegalArgumentException(String.format(
                "Input must contain exactly %d features: %s. Got %d features: %s",
                featureNames.length, Arrays.toString(featureNames),
                input.size(), input.keySet()));
        }
        
        float[] modelInput = new float[featureNames.length];
        
        for (int i = 0; i < featureNames.length; i++) {
            String featureName = featureNames[i];
            Object value = input.get(featureName);
            
            if (value == null) {
                throw new IllegalArgumentException("Missing required feature: " + featureName);
            }
            
            Feature feature = features[i];
            switch (feature.getType()) {
                case EMBEDDING:
                case ONEHOT:
                    Dictionary dict = featureDictionaries.get(featureName);
                    try {
                        int index = dict.getIndex(value);
                        modelInput[i] = (float) index;
                    } catch (IllegalStateException e) {
                        // Dictionary is full - provide helpful error message
                        if (!features[i].isLRU() && e.getMessage().contains("Dictionary is full")) {
                            throw new IllegalStateException(String.format(
                                "Dictionary for feature '%s' has grown to %d entries, exceeding maxUniqueValues=%d. " +
                                "This indicates unbounded vocabulary growth. Consider: " +
                                "1) Using Feature.hashedEmbedding() instead for high-cardinality features, " +
                                "2) Pre-processing data to limit unique values, or " +
                                "3) Using Feature.embeddingLRU() or Feature.oneHotLRU() for online learning.",
                                featureName, dict.size(), features[i].getMaxUniqueValues()));
                        }
                        throw e; // Re-throw if it's a different error
                    }
                    break;
                    
                case PASSTHROUGH:
                case AUTO_NORMALIZE:
                case SCALE_BOUNDED:
                    if (value instanceof Number) {
                        modelInput[i] = ((Number) value).floatValue();
                    } else {
                        throw new IllegalArgumentException(String.format(
                            "Feature '%s' (%s) requires numerical value but received: %s",
                            featureName, feature.getType(), value.getClass().getSimpleName()));
                    }
                    break;
            }
        }
        
        return modelInput;
    }
    
    /**
     * Validate that Map input is allowed for this model.
     * Subclasses can override to add specific validation.
     */
    protected void validateMapInput() {
        // Default implementation - no additional validation
    }
    
    // Abstract methods that subclasses must implement
    protected abstract void trainInternal(Object input, float[] targets);
    protected abstract float[] predictInternal(Object input);
    
    /**
     * Get the loss function to use for training.
     * Subclasses must specify their appropriate loss function.
     */
    protected abstract Loss getLossFunction();
    
    // ===============================
    // BULK TRAINING METHODS
    // ===============================
    
    /**
     * Train on streaming data from an iterator with separate validation data.
     * This allows training on datasets larger than memory with full metrics support.
     * 
     * <p><b>Example - Stream training data with validation:</b>
     * <pre>{@code
     * // Training iterator from large file
     * DataIterator<String[], String> trainData = DataIterator.fromFile(
     *     "train_data.txt", 
     *     line -> {
     *         String[] parts = line.split("\t");
     *         return DataBatch.single(
     *             Arrays.copyOf(parts, parts.length - 1),
     *             parts[parts.length - 1]
     *         );
     *     }
     * );
     * 
     * // Validation data (can be in-memory since usually smaller)
     * List<String[]> valSequences = loadValidationSequences();
     * List<String> valTargets = loadValidationTargets();
     * 
     * // Train with full metrics
     * model.trainBulk(trainData, valSequences, valTargets, config);
     * }</pre>
     * 
     * @param trainIterator iterator providing training data batches
     * @param valInputs validation inputs (can be null for no validation)
     * @param valTargets validation targets (can be null for no validation)
     * @param config training configuration
     * @return training result with full metrics
     */
    public SimpleNetTrainingResult trainBulk(DataIterator<?, T> trainIterator,
                                           List<?> valInputs, List<T> valTargets,
                                           SimpleNetTrainingConfig config) {
        // Encode validation data once
        float[][] encodedValInputs = null;
        float[][] encodedValTargets = null;
        
        if (valInputs != null && valTargets != null) {
            if (valInputs.size() != valTargets.size()) {
                throw new IllegalArgumentException(
                    "Validation inputs and targets must have same size");
            }
            
            encodedValInputs = new float[valInputs.size()][];
            for (int i = 0; i < valInputs.size(); i++) {
                Object input = valInputs.get(i);
                if (input instanceof Map) {
                    @SuppressWarnings("unchecked")
                    Map<String, Object> mapInput = (Map<String, Object>) input;
                    encodedValInputs[i] = convertFromMap(mapInput);
                } else if (input instanceof float[]) {
                    encodedValInputs[i] = convertFromFloatArray((float[]) input);
                } else {
                    throw new IllegalArgumentException(
                        "Unsupported validation input type: " + input.getClass().getSimpleName());
                }
            }
            
            encodedValTargets = encodeTargets(valTargets);
        }
        
        // Create BatchTrainer with appropriate loss function
        BatchTrainer trainer = new BatchTrainer(
            underlyingNet, 
            getLossFunction(),
            config.getBatchConfig()
        );
        
        // Build callbacks
        List<TrainingCallback> callbacks = buildCallbacks(config, trainer);
        for (TrainingCallback callback : callbacks) {
            trainer.withCallback(callback);
        }
        
        // Prepare for training
        long startTimeNanos = System.nanoTime();
        int epochs = config.getBatchConfig().epochs;
        TrainingMetrics metrics = new TrainingMetrics();
        
        try (trainIterator) {
            // Training loop - each full pass through iterator is one epoch
            for (int epoch = 0; epoch < epochs; epoch++) {
                // Reset iterator for new epoch (except first)
                if (epoch > 0) {
                    trainIterator.reset();
                }
                
                // Track epoch metrics
                double epochLoss = 0;
                double epochAccuracy = 0;
                int epochSamples = 0;
                
                // Process all training data in this epoch
                while (trainIterator.hasNext()) {
                    // Get next batch
                    DataBatch<?, T> batch = trainIterator.nextBatch(config.getBatchConfig().batchSize);
                    
                    // Convert and encode batch
                    float[][] batchInputs = new float[batch.size()][];
                    for (int i = 0; i < batch.size(); i++) {
                        Object input = batch.getInputs()[i];
                        if (input instanceof Map) {
                            @SuppressWarnings("unchecked")
                            Map<String, Object> mapInput = (Map<String, Object>) input;
                            batchInputs[i] = convertFromMap(mapInput);
                        } else if (input instanceof float[]) {
                            batchInputs[i] = convertFromFloatArray((float[]) input);
                        } else {
                            throw new IllegalArgumentException(
                                "Unsupported input type: " + input.getClass().getSimpleName());
                        }
                    }
                    
                    // Encode targets from batch
                    @SuppressWarnings("unchecked")
                    T[] targetsArray = (T[]) batch.getTargets();
                    List<T> targetsList = Arrays.asList(targetsArray);
                    float[][] batchTargets = encodeTargets(targetsList);
                    
                    // Train this batch and collect metrics
                    underlyingNet.trainBatch(batchInputs, batchTargets);
                    
                    // Calculate batch metrics for tracking
                    // Note: This adds overhead but provides proper metrics
                    Loss loss = getLossFunction();
                    for (int i = 0; i < batchInputs.length; i++) {
                        float[] prediction = underlyingNet.predict(batchInputs[i]);
                        epochLoss += loss.loss(prediction, batchTargets[i]);
                        
                        // For classification tasks
                        int predictedClass = Utils.argmax(prediction);
                        int actualClass = Utils.argmax(batchTargets[i]);
                        if (predictedClass == actualClass) {
                            epochAccuracy += 1.0;
                        }
                    }
                    
                    epochSamples += batch.size();
                }
                
                // Evaluate on validation set if provided
                double valLoss = 0;
                double valAccuracy = 0;
                if (encodedValInputs != null) {
                    // Run validation
                    for (int i = 0; i < encodedValInputs.length; i++) {
                        float[] prediction = underlyingNet.predict(encodedValInputs[i]);
                        
                        // Calculate loss for this sample
                        Loss loss = getLossFunction();
                        valLoss += loss.loss(prediction, encodedValTargets[i]);
                        
                        // Calculate accuracy (for classification)
                        int predictedClass = Utils.argmax(prediction);
                        int actualClass = Utils.argmax(encodedValTargets[i]);
                        if (predictedClass == actualClass) {
                            valAccuracy += 1.0;
                        }
                    }
                    valLoss /= encodedValInputs.length;
                    valAccuracy /= encodedValInputs.length;
                }
                
                // Record epoch metrics
                metrics.recordEpoch(
                    epoch + 1,  // 1-based epoch number
                    epochLoss / epochSamples,  // average training loss
                    epochAccuracy / epochSamples,  // average training accuracy  
                    valLoss,
                    valAccuracy,
                    epochSamples  // samples seen in this epoch
                );
                
                // Notify callbacks about epoch completion
                for (TrainingCallback callback : callbacks) {
                    callback.onEpochEnd(epoch + 1, metrics);
                }
                
                // Check if training should stop (early stopping, etc.)
                if (trainer.getStopFlag().get()) {
                    break;
                }
            }
        } catch (UnsupportedOperationException e) {
            throw new IllegalArgumentException(
                "Iterator does not support reset() - required for multi-epoch training. " +
                "Use a single epoch or provide an iterator with reset support.", e);
        } catch (Exception e) {
            throw new RuntimeException("Error during iterator-based training", e);
        }
        
        long trainingTimeNanos = System.nanoTime() - startTimeNanos;
        long trainingTimeMs = trainingTimeNanos / 1_000_000;
        
        // Create result with collected metrics
        BatchTrainer.TrainingResult batchResult = new BatchTrainer.TrainingResult(
            metrics,
            trainer
        );
        
        return new SimpleNetTrainingResult(
            batchResult,
            trainingTimeMs,
            epochs
        );
    }
    
    /**
     * Train on streaming data from an iterator.
     * Uses automatic validation split from the iterator data.
     * Note: This requires loading data to perform the split, so may not be suitable
     * for truly massive datasets.
     * 
     * @param dataIterator iterator providing training data batches
     * @param config training configuration (validationSplit will be applied)
     * @return training result with metrics
     */
    public SimpleNetTrainingResult trainBulk(DataIterator<?, T> dataIterator,
                                            SimpleNetTrainingConfig config) {
        // When no validation data provided, collect all data for standard training
        // This ensures proper metrics and validation split handling
        List<Object> allInputs = new ArrayList<>();
        List<T> allTargets = new ArrayList<>();
        
        try (dataIterator) {
            while (dataIterator.hasNext()) {
                DataBatch<?, T> batch = dataIterator.nextBatch(config.getBatchConfig().batchSize);
                
                for (int i = 0; i < batch.size(); i++) {
                    allInputs.add(batch.getInputs()[i]);
                    allTargets.add(batch.getTargets()[i]);
                }
            }
        } catch (Exception e) {
            throw new RuntimeException("Error loading data from iterator", e);
        }
        
        if (allInputs.isEmpty()) {
            return new SimpleNetTrainingResult(null, 0, 0);
        }
        
        // Use standard training with collected data
        return trainBulk(allInputs, allTargets, config);
    }
    
    /**
     * Train on a batch of samples.
     * This method handles both Map-based inputs (for mixed feature models) and 
     * array-based inputs (for simple models).
     * 
     * @param inputs list of inputs - either Map<String, Object> or float[] depending on model type
     * @param targets list of target values matching the input order
     * @param config training configuration
     * @return training result with metrics
     */
    public SimpleNetTrainingResult trainBulk(List<?> inputs, List<T> targets, 
                                            SimpleNetTrainingConfig config) {
        if (inputs.size() != targets.size()) {
            throw new IllegalArgumentException("Inputs and targets must have the same size");
        }
        
        if (inputs.isEmpty()) {
            return trainWithEncodedData(new float[0][], new float[0][], config);
        }
        
        // Determine input type from first element
        Object firstInput = inputs.get(0);
        
        if (firstInput instanceof Map) {
            // Map-based inputs
            if (!usesFeatureMapping) {
                throw new IllegalArgumentException(
                    "This model does not use mixed features. Use List<float[]> instead.");
            }
            
            float[][] encodedInputs = new float[inputs.size()][];
            
            for (int i = 0; i < inputs.size(); i++) {
                @SuppressWarnings("unchecked")
                Map<String, Object> mapInput = (Map<String, Object>) inputs.get(i);
                encodedInputs[i] = convertFromMap(mapInput);
            }
            
            // Let subclass encode targets
            float[][] encodedTargets = encodeTargets(targets);
            
            return trainWithEncodedData(encodedInputs, encodedTargets, config);
            
        } else if (firstInput instanceof float[]) {
            // Array-based inputs - work for both simple and mixed feature models
            float[][] encodedInputs = new float[inputs.size()][];
            
            for (int i = 0; i < inputs.size(); i++) {
                float[] arrayInput = (float[]) inputs.get(i);
                encodedInputs[i] = convertFromFloatArray(arrayInput);
            }
            
            // Let subclass encode targets
            float[][] encodedTargets = encodeTargets(targets);
            
            return trainWithEncodedData(encodedInputs, encodedTargets, config);
            
        } else {
            throw new IllegalArgumentException(
                "Inputs must be either Map<String, Object> or float[]. Got: " + 
                firstInput.getClass().getSimpleName());
        }
    }
    
    /**
     * Encode target values to float arrays.
     * Subclasses implement this to handle their specific target types.
     * 
     * @param targets list of target values
     * @return encoded targets as float arrays
     */
    protected abstract float[][] encodeTargets(List<T> targets);
    
    // ===============================
    // SHARED TRAINING IMPLEMENTATION
    // ===============================
    
    /**
     * Common implementation for training with encoded data and validation split.
     * This method is shared by all SimpleNet implementations to avoid code duplication.
     * 
     * @param trainInputs pre-encoded training input arrays
     * @param trainTargets pre-encoded training target arrays
     * @param valInputs pre-encoded validation input arrays (can be null)
     * @param valTargets pre-encoded validation target arrays (can be null)
     * @param config training configuration
     * @return training result with metrics
     */
    protected SimpleNetTrainingResult trainWithEncodedData(float[][] trainInputs, float[][] trainTargets,
                                                          float[][] valInputs, float[][] valTargets,
                                                          SimpleNetTrainingConfig config) {
        // Create BatchTrainer with appropriate loss function FIRST
        BatchTrainer trainer = new BatchTrainer(
            underlyingNet, 
            getLossFunction(),
            config.getBatchConfig()
        );
        
        // Build callbacks AFTER creating trainer so we can pass its stopFlag
        List<TrainingCallback> callbacks = buildCallbacks(config, trainer);
        
        // Add callbacks
        for (TrainingCallback callback : callbacks) {
            trainer.withCallback(callback);
        }
        
        // Train
        long startTimeNanos = System.nanoTime();
        BatchTrainer.TrainingResult batchResult;
        
        if (valInputs != null && valTargets != null)
            batchResult = trainer.fit(trainInputs, trainTargets, valInputs, valTargets);
        else
            batchResult = trainer.fit(trainInputs, trainTargets);
        
        long trainingTimeNanos = System.nanoTime() - startTimeNanos;
        long trainingTimeMs = trainingTimeNanos / 1_000_000; // Convert to milliseconds
        
        // Ensure we always record at least 1ms (training can never take 0 time)
        if (trainingTimeMs == 0 && trainingTimeNanos > 0) {
            trainingTimeMs = 1;
        }
        
        return new SimpleNetTrainingResult(
            batchResult, 
            trainingTimeMs, 
            batchResult.getMetrics().getEpochCount()
        );
    }
    
    /**
     * Common implementation for training with encoded data.
     * This method is shared by all SimpleNet implementations to avoid code duplication.
     * 
     * @param encodedInputs pre-encoded input arrays
     * @param encodedTargets pre-encoded target arrays
     * @param config training configuration
     * @return training result with metrics
     */
    protected SimpleNetTrainingResult trainWithEncodedData(float[][] encodedInputs, float[][] encodedTargets,
                                                          SimpleNetTrainingConfig config) {
        return trainWithEncodedData(encodedInputs, encodedTargets, null, null, config);
    }
    
    /**
     * Build training callbacks based on configuration.
     * 
     * @param config training configuration
     * @param trainer the BatchTrainer instance to get the stopFlag from
     * @return list of callbacks to use during training
     */
    protected List<TrainingCallback> buildCallbacks(SimpleNetTrainingConfig config, BatchTrainer trainer) {
        List<TrainingCallback> callbacks = new ArrayList<>();
        
        if (config.isEarlyStoppingEnabled()) {
            callbacks.add(new EarlyStoppingCallback(
                config.getEarlyStoppingPatience(),
                config.getEarlyStoppingMinDelta(),
                trainer.getStopFlag()  // Use the trainer's stopFlag instead of creating a new one
            ));
        }
        
        if (config.isCheckpointingEnabled()) {
            String monitorMetric = getCheckpointMonitorMetric();
            // Use the new generic checkpoint callback to save the full model type
            callbacks.add(new ModelCheckpointCallback.WithSerializableModel<>(
                this,  // Save the full SimpleNet subclass, not just the NeuralNet
                config.getCheckpointPath(),
                monitorMetric,
                config.isCheckpointOnlyBest(),
                0
            ));
        }
        
        if (config.isVisualizationEnabled()) {
            callbacks.add(new VisualizationCallback(config.getVisualizationPath()));
        }
        
        return callbacks;
    }
    
    /**
     * Get the metric to monitor for model checkpointing.
     * Subclasses can override to customize the monitored metric.
     * 
     * @return metric name to monitor
     */
    protected String getCheckpointMonitorMetric() {
        // Default to monitoring validation accuracy for classification
        // and validation loss for regression
        return "val_accuracy";
    }
    
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
        
        if (!(outputLayer instanceof RegressionOutput)) {
            throw new IllegalArgumentException(
                "For regression, use a regression output layer (LinearRegression, HuberRegression, etc.). " +
                "Found: " + outputLayer.getClass().getSimpleName());
        }
        
        if (outputLayer.getOutputSize() != 1) {
            throw new IllegalArgumentException(
                "For single regression, use a regression output with 1 output. " +
                "Found output size: " + outputLayer.getOutputSize() + 
                ". Use ofMultiFloatRegression() for multiple outputs.");
        }
    }
    
    private static void validateMultiRegressionNetwork(NeuralNet net) {
        Layer outputLayer = net.getOutputLayer();
        
        if (!(outputLayer instanceof RegressionOutput)) {
            throw new IllegalArgumentException(
                "For regression, use a regression output layer (LinearRegression, HuberRegression, etc.). " +
                "Found: " + outputLayer.getClass().getSimpleName());
        }
        
        if (outputLayer.getOutputSize() < 1) {
            throw new IllegalArgumentException(
                "For multi-output regression, use a regression output with n > 0 outputs. " +
                "Found output size: " + outputLayer.getOutputSize());
        }
    }
    
}