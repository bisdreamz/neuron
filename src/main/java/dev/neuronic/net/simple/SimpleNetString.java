package dev.neuronic.net.simple;

import dev.neuronic.net.NeuralNet;
import dev.neuronic.net.SamplingConfig;
import dev.neuronic.net.Dictionary;
import dev.neuronic.net.common.Utils;
import dev.neuronic.net.layers.Feature;
import dev.neuronic.net.layers.Layer;
import dev.neuronic.net.layers.MixedFeatureInputLayer;
import dev.neuronic.net.layers.InputSequenceEmbeddingLayer;
import dev.neuronic.net.serialization.SerializationConstants;
import dev.neuronic.net.training.BatchTrainer;
import dev.neuronic.net.training.TrainingCallback;
import dev.neuronic.net.training.EarlyStoppingCallback;
import dev.neuronic.net.training.ModelCheckpointCallback;
import dev.neuronic.net.training.VisualizationCallback;
import dev.neuronic.net.losses.CrossEntropyLoss;

import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.IOException;
import java.nio.file.Path;
import java.util.*;
import java.util.concurrent.Executors;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ExecutorService;
import java.util.ArrayList;

/**
 * Type-safe neural network wrapper for string classification (e.g., sentiment analysis, text classification).
 * 
 * <p><b>What it does:</b> Handles classification tasks where labels are strings and you want
 * a string result representing the predicted class name.
 * 
 * <p><b>Perfect for:</b>
 * <ul>
 *   <li><b>Sentiment analysis:</b> Returns "positive", "negative", "neutral"</li>
 *   <li><b>Text classification:</b> Returns "spam", "legitimate"</li>
 *   <li><b>Document categorization:</b> Returns category names</li>
 *   <li><b>Intent recognition:</b> Returns intent names</li>
 * </ul>
 * 
 * <p><b>Key Benefits:</b>
 * <ul>
 *   <li><b>Type safety:</b> Returns String - no casting needed</li>
 *   <li><b>Human readable:</b> Returns meaningful labels instead of numbers</li>
 *   <li><b>Automatic dictionary:</b> Builds label mapping during training</li>
 *   <li><b>Label preservation:</b> Returns original string labels from training</li>
 * </ul>
 * 
 * <p><b>Example - Sentiment Analysis:</b>
 * <pre>{@code
 * // Create neural network for sentiment classification
 * NeuralNet net = NeuralNet.newBuilder()
 *     .input(3)
 *     .layer(Layers.inputMixed(optimizer,
 *         Feature.embedding(10000, 128),  // text_tokens
 *         Feature.oneHot(4),              // source_type
 *         Feature.passthrough()           // text_length
 *     ))
 *     .layer(Layers.hiddenDenseRelu(256))
 *     .output(Layers.outputSoftmaxCrossEntropy(3));
 * 
 * // Create type-safe classifier
 * SimpleNetString classifier = SimpleNet.ofStringClassification(net);
 * 
 * // Train with string labels - automatic label discovery
 * classifier.train(Map.of(
 *     "text_tokens", "This product is amazing!",
 *     "source_type", 1,              // 1=customer review
 *     "text_length", 127.5f
 * ), "positive");  // String label
 * 
 * classifier.train(Map.of(
 *     "text_tokens", "Terrible service",
 *     "source_type", 1,
 *     "text_length", 89.0f
 * ), "negative");
 * 
 * // Predict - returns String (no casting!)
 * String sentiment = classifier.predict(Map.of(
 *     "text_tokens", "Great customer support!",
 *     "source_type", 1,
 *     "text_length", 145.0f
 * ));
 * System.out.println("Sentiment: " + sentiment);  // "Sentiment: positive"
 * }</pre>
 * 
 * <p><b>Thread Safety:</b> All methods are thread-safe for concurrent training and prediction.
 */
public class SimpleNetString extends SimpleNet {
    
    private final Dictionary labelDictionary;
    private final boolean usesFeatureMapping;
    private final String[] featureNames;
    private final ConcurrentHashMap<String, Dictionary> featureDictionaries;
    private final Feature[] features;
    private volatile SamplingConfig samplingConfig = SamplingConfig.argmax();
    
    /**
     * Create a SimpleNetString without output names (for backward compatibility and deserialization).
     * Package-private.
     */
    SimpleNetString(NeuralNet underlyingNet) {
        this(underlyingNet, null);
    }
    
    /**
     * Create a SimpleNetString for string classification.
     * Package-private - use SimpleNet.ofStringClassification() instead.
     */
    SimpleNetString(NeuralNet underlyingNet, Set<String> outputNames) {
        super(underlyingNet, outputNames);
        this.labelDictionary = new Dictionary();
        
        // Auto-detect if this uses mixed features or simple arrays
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
                    featureDictionaries.put(featureNames[i], new Dictionary());
                }
            }
        } else {
            this.usesFeatureMapping = false;
            this.features = null;
            this.featureNames = null;
            this.featureDictionaries = null;
        }
    }
    
    /**
     * Train the classifier with a single example.
     * 
     * <p><b>For mixed features:</b>
     * <pre>{@code
     * classifier.train(Map.of(
     *     "feature1", "some_string",
     *     "feature2", 42.5f
     * ), "positive");  // Mixed features + string label
     * }</pre>
     * 
     * <p><b>For raw arrays:</b>
     * <pre>{@code
     * classifier.train(textEmbeddings, "spam");  // float[] + string label
     * }</pre>
     * 
     * @param input either float[] for raw arrays or Map&lt;String, Object&gt; for mixed features
     * @param label string class label (e.g., "positive", "spam", "urgent")
     */
    public void train(Object input, String label) {
        float[] modelInput = convertInput(input);
        
        // Add label to dictionary if not seen before
        int classIndex = labelDictionary.getIndex(label);
        
        // Convert label to one-hot or appropriate format
        float[] modelTarget = createTargetVector(classIndex);
        
        underlyingNet.train(modelInput, modelTarget);
    }
    
    /**
     * Predict the class for new input.
     * 
     * @param input either float[] for raw arrays or Map&lt;String, Object&gt; for mixed features  
     * @return predicted string class label (same type as used in training)
     */
    public String predictString(Object input) {
        float[] modelInput = convertInput(input);
        
        // Apply sampling strategy using new prediction methods
        int predictedClassIndex = applySamplingStrategy(modelInput);
        
        // Always use the label dictionary for classification
        // InputSequenceEmbeddingLayer is for input processing, not output labels
        Object originalLabel = labelDictionary.getValue(predictedClassIndex);
        return originalLabel != null ? (String) originalLabel : "class_" + predictedClassIndex;
    }
    
    /**
     * Apply the configured sampling strategy to select a class.
     */
    private int applySamplingStrategy(float[] modelInput) {
        switch (samplingConfig.getStrategy()) {
            case ARGMAX:
                return (int) underlyingNet.predictArgmax(modelInput);
            case TEMPERATURE:
                return (int) underlyingNet.predictWithTemperature(modelInput, samplingConfig.getTemperature());
            case TOP_K:
                return (int) underlyingNet.predictSampleTopK(modelInput, samplingConfig.getK(), samplingConfig.getTemperature());
            case TOP_P:
                return (int) underlyingNet.predictSampleTopP(modelInput, samplingConfig.getP(), samplingConfig.getTemperature());
            default:
                return (int) underlyingNet.predictArgmax(modelInput);
        }
    }
    
    /**
     * Get the top K predicted classes with their labels.
     * 
     * @param input input data
     * @param k number of top predictions to return
     * @return array of top k class predictions as strings, sorted by confidence (highest first)
     */
    public String[] predictTopK(Object input, int k) {
        float[] modelInput = convertInput(input);
        float[] topKIndices = underlyingNet.predictTopK(modelInput, k);
        
        // Convert float indices to String labels
        String[] topKLabels = new String[topKIndices.length];
        for (int i = 0; i < topKIndices.length; i++) {
            int classIndex = (int) topKIndices[i];
            Object originalLabel = labelDictionary.getValue(classIndex);
            topKLabels[i] = originalLabel != null ? (String) originalLabel : "class_" + classIndex;
        }
        
        return topKLabels;
    }
    
    /**
     * Get the confidence (probability) for the predicted class.
     * 
     * @param input input data
     * @return confidence score between 0.0 and 1.0 for the top predicted class
     */
    public float predictConfidence(Object input) {
        float[] modelInput = convertInput(input);
        float[] probabilities = underlyingNet.predict(modelInput);
        
        int predictedClass = Utils.argmax(probabilities);
        return probabilities[predictedClass];
    }
    
    /**
     * Get the number of unique classes seen during training.
     */
    public int getClassCount() {
        return labelDictionary.size();
    }
    
    /**
     * Check if a specific label has been seen during training.
     */
    public boolean hasSeenLabel(String label) {
        return labelDictionary.containsValue(label);
    }
    
    /**
     * Train the network with multiple examples using bulk training features.
     * Supports epochs, validation split, and callbacks.
     * 
     * @param inputs list of inputs (float[], Map<String, Object>, or String[])
     * @param labels list of string labels
     * @param config training configuration
     * @return training result with metrics
     */
    public SimpleNetTrainingResult trainBulk(List<Object> inputs, List<String> labels, 
                                           SimpleNetTrainingConfig config) {
        if (inputs.size() != labels.size()) {
            throw new IllegalArgumentException("Inputs and labels must have the same size");
        }
        
        // Check if we're using InputSequenceEmbeddingLayer
        Layer inputLayer = underlyingNet.getInputLayer();
        if (inputLayer instanceof InputSequenceEmbeddingLayer) {
            // Special handling for sequence embedding - use custom training
            return trainBulkSequenceEmbedding(inputs, labels, null, null, config);
        }
        
        // Convert to arrays for BatchTrainer
        float[][] encodedInputs = new float[inputs.size()][];
        float[][] encodedTargets = new float[labels.size()][];
        
        for (int i = 0; i < inputs.size(); i++) {
            encodedInputs[i] = convertInput(inputs.get(i));
            encodedTargets[i] = createTargetVector(getLabelIndex(labels.get(i)));
        }
        
        // Build callbacks
        List<TrainingCallback> callbacks = buildCallbacks(config);
        
        // Create BatchTrainer for classification
        BatchTrainer trainer = new BatchTrainer(underlyingNet, CrossEntropyLoss.INSTANCE, config.getBatchConfig());
        
        // Add callbacks
        for (TrainingCallback callback : callbacks) {
            trainer.withCallback(callback);
        }
        
        // Train using BatchTrainer
        long startTime = System.currentTimeMillis();
        BatchTrainer.TrainingResult batchResult = trainer.fit(
            encodedInputs, encodedTargets);
        long trainingTime = System.currentTimeMillis() - startTime;
        
        return new SimpleNetTrainingResult(
            batchResult, 
            trainingTime, 
            batchResult.getMetrics().getEpochCount()
        );
    }
    
    /**
     * Train with pre-split train and validation data.
     * 
     * @param trainInputs training inputs
     * @param trainLabels training labels
     * @param valInputs validation inputs
     * @param valLabels validation labels
     * @param config training configuration
     * @return training result with metrics
     */
    public SimpleNetTrainingResult trainBulk(List<Object> trainInputs, List<String> trainLabels,
                                           List<Object> valInputs, List<String> valLabels,
                                           SimpleNetTrainingConfig config) {
        // Validate inputs
        if (trainInputs.size() != trainLabels.size())
            throw new IllegalArgumentException("Training inputs and labels must have the same size");
        
        if (valInputs != null && valLabels != null && valInputs.size() != valLabels.size())
            throw new IllegalArgumentException("Validation inputs and labels must have the same size");
        
        // Check if using sequence embedding
        Layer inputLayer = underlyingNet.getInputLayer();
        if (inputLayer instanceof InputSequenceEmbeddingLayer) {
            return trainBulkSequenceEmbedding(trainInputs, trainLabels, valInputs, valLabels, config);
        }
        
        // Convert training data
        float[][] encodedTrainInputs = new float[trainInputs.size()][];
        float[][] encodedTrainTargets = new float[trainLabels.size()][];
        
        for (int i = 0; i < trainInputs.size(); i++) {
            encodedTrainInputs[i] = convertInput(trainInputs.get(i));
            encodedTrainTargets[i] = createTargetVector(getLabelIndex(trainLabels.get(i)));
        }
        
        // Convert validation data if provided
        float[][] encodedValInputs = null;
        float[][] encodedValTargets = null;
        
        if (valInputs != null && valLabels != null) {
            encodedValInputs = new float[valInputs.size()][];
            encodedValTargets = new float[valLabels.size()][];
            
            for (int i = 0; i < valInputs.size(); i++) {
                encodedValInputs[i] = convertInput(valInputs.get(i));
                encodedValTargets[i] = createTargetVector(getLabelIndex(valLabels.get(i)));
            }
        }
        
        // Build callbacks
        List<TrainingCallback> callbacks = buildCallbacks(config);
        
        // Create BatchTrainer for classification
        BatchTrainer trainer = new BatchTrainer(underlyingNet, CrossEntropyLoss.INSTANCE, config.getBatchConfig());
        
        // Add callbacks
        for (TrainingCallback callback : callbacks) {
            trainer.withCallback(callback);
        }
        
        // Train using BatchTrainer with pre-split data
        long startTime = System.currentTimeMillis();
        BatchTrainer.TrainingResult batchResult = trainer.fit(
            encodedTrainInputs, encodedTrainTargets, 
            encodedValInputs, encodedValTargets);
        long trainingTime = System.currentTimeMillis() - startTime;
        
        return new SimpleNetTrainingResult(
            batchResult, 
            trainingTime, 
            batchResult.getMetrics().getEpochCount()
        );
    }
    
    /**
     * Train using arrays (convenience method).
     */
    public SimpleNetTrainingResult trainBulk(Object[] inputs, String[] labels,
                                           SimpleNetTrainingConfig config) {
        return trainBulk(Arrays.asList(inputs), Arrays.asList(labels), config);
    }
    
    private List<TrainingCallback> buildCallbacks(SimpleNetTrainingConfig config) {
        List<TrainingCallback> callbacks = new ArrayList<>();
        
        if (config.isEarlyStoppingEnabled()) {
            callbacks.add(new EarlyStoppingCallback(
                config.getEarlyStoppingPatience(),
                config.getEarlyStoppingMinDelta(),
                new java.util.concurrent.atomic.AtomicBoolean()
            ));
        }
        
        if (config.isCheckpointingEnabled()) {
            callbacks.add(new ModelCheckpointCallback.WithModel(
                underlyingNet,
                config.getCheckpointPath(),
                "val_accuracy",  // For classification, monitor validation accuracy
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
     * Special bulk training for InputSequenceEmbeddingLayer that handles string sequences properly.
     */
    private SimpleNetTrainingResult trainBulkSequenceEmbedding(List<Object> trainInputs, List<String> trainLabels,
                                                              List<Object> valInputs, List<String> valLabels,
                                                              SimpleNetTrainingConfig config) {
        InputSequenceEmbeddingLayer embeddingLayer = (InputSequenceEmbeddingLayer) underlyingNet.getInputLayer();
        
        // Convert training sequences to token IDs
        float[][] tokenizedTrainInputs = new float[trainInputs.size()][];
        float[][] encodedTrainTargets = new float[trainLabels.size()][];
        
        // Use parallel processing if input size is large
        if (trainInputs.size() > 100) {
            // Parallel tokenization for large datasets
            int numThreads = Runtime.getRuntime().availableProcessors();
            ExecutorService executor = Executors.newFixedThreadPool(numThreads);
            int chunkSize = (trainInputs.size() + numThreads - 1) / numThreads;
            
            List<java.util.concurrent.Future<?>> futures = new ArrayList<>();
            
            for (int t = 0; t < numThreads; t++) {
                final int startIdx = t * chunkSize;
                final int endIdx = Math.min(startIdx + chunkSize, trainInputs.size());
                
                futures.add(executor.submit(() -> {
                    for (int i = startIdx; i < endIdx; i++) {
                        String[] sequence = convertToStringArray(trainInputs.get(i));
                        tokenizedTrainInputs[i] = tokenizeSequence(sequence, embeddingLayer);
                        
                        encodedTrainTargets[i] = createTargetVector(getLabelIndex(trainLabels.get(i)));
                    }
                }));
            }
            
            // Wait for all tokenization to complete
            for (java.util.concurrent.Future<?> future : futures) {
                try {
                    future.get();
                } catch (Exception e) {
                    e.printStackTrace();
                    System.exit(1);
                }
            }
            
            executor.shutdown();
        } else {
            // Sequential tokenization for small datasets
            for (int i = 0; i < trainInputs.size(); i++) {
                String[] sequence = convertToStringArray(trainInputs.get(i));
                tokenizedTrainInputs[i] = tokenizeSequence(sequence, embeddingLayer);
                
                encodedTrainTargets[i] = createTargetVector(getLabelIndex(trainLabels.get(i)));
            }
        }
        
        // Convert validation sequences if provided
        float[][] tokenizedValInputs = null;
        float[][] encodedValTargets = null;
        
        if (valInputs != null && valLabels != null) {
            tokenizedValInputs = new float[valInputs.size()][];
            encodedValTargets = new float[valLabels.size()][];
            
            // Make arrays final for lambda access
            final float[][] finalTokenizedValInputs = tokenizedValInputs;
            final float[][] finalEncodedValTargets = encodedValTargets;
            
            // Process validation data (can be parallelized if large)
            if (valInputs.size() > 100) {
                int numThreads = Runtime.getRuntime().availableProcessors();
                ExecutorService executor = Executors.newFixedThreadPool(numThreads);
                int chunkSize = (valInputs.size() + numThreads - 1) / numThreads;
                
                List<java.util.concurrent.Future<?>> futures = new ArrayList<>();
                
                for (int t = 0; t < numThreads; t++) {
                    final int startIdx = t * chunkSize;
                    final int endIdx = Math.min(startIdx + chunkSize, valInputs.size());
                    
                    futures.add(executor.submit(() -> {
                        for (int i = startIdx; i < endIdx; i++) {
                            String[] sequence = convertToStringArray(valInputs.get(i));
                            finalTokenizedValInputs[i] = tokenizeSequence(sequence, embeddingLayer);
                            
                            finalEncodedValTargets[i] = createTargetVector(getLabelIndex(valLabels.get(i)));
                        }
                    }));
                }
                
                for (java.util.concurrent.Future<?> future : futures) {
                    try {
                        future.get();
                    } catch (Exception e) {
                        e.printStackTrace();
                        System.exit(1);
                    }
                }
                
                executor.shutdown();
            } else {
                for (int i = 0; i < valInputs.size(); i++) {
                    String[] sequence = convertToStringArray(valInputs.get(i));
                    tokenizedValInputs[i] = tokenizeSequence(sequence, embeddingLayer);
                    
                    encodedValTargets[i] = createTargetVector(getLabelIndex(valLabels.get(i)));
                }
            }
        }
        
        // Build callbacks
        List<TrainingCallback> callbacks = buildCallbacks(config);
        
        // Create BatchTrainer for classification (CrossEntropy loss)
        BatchTrainer trainer = new BatchTrainer(
            underlyingNet, 
            CrossEntropyLoss.INSTANCE,
            config.getBatchConfig()
        );
        
        // Add callbacks
        for (TrainingCallback callback : callbacks)
            trainer.withCallback(callback);
        
        // Train with pre-split data
        long startTime = System.currentTimeMillis();
        BatchTrainer.TrainingResult batchResult = trainer.fit(
            tokenizedTrainInputs, encodedTrainTargets, 
            tokenizedValInputs, encodedValTargets
        );
        long trainingTime = System.currentTimeMillis() - startTime;
        
        return new SimpleNetTrainingResult(
            batchResult, 
            trainingTime, 
            batchResult.getMetrics().getEpochCount()
        );
    }
    
    /**
     * Convert a string sequence to token IDs using the embedding layer's vocabulary.
     */
    private float[] tokenizeSequence(String[] sequence, InputSequenceEmbeddingLayer embeddingLayer) {
        float[] tokenIds = new float[sequence.length];
        for (int i = 0; i < sequence.length; i++) {
            tokenIds[i] = embeddingLayer.getTokenId(sequence[i]);
        }
        return tokenIds;
    }
    
    private String[] convertToStringArray(Object input) {
        if (input instanceof String[]) {
            return (String[]) input;
        } else if (input instanceof List) {
            @SuppressWarnings("unchecked")
            List<String> list = (List<String>) input;
            return list.toArray(new String[0]);
        } else {
            throw new IllegalArgumentException(
                "For sequence embedding models, input must be String[] or List<String>, got: " + 
                (input == null ? "null" : input.getClass().getSimpleName()));
        }
    }
    
    // Private helper methods
    
    public float[] convertInput(Object input) {
        // Check if this is a sequence embedding layer
        Layer inputLayer = underlyingNet.getInputLayer();
        if (inputLayer instanceof InputSequenceEmbeddingLayer) {
            // For single example training, tokenize the sequence
            InputSequenceEmbeddingLayer embeddingLayer = (InputSequenceEmbeddingLayer) inputLayer;
            String[] sequence = convertToStringArray(input);
            return tokenizeSequence(sequence, embeddingLayer);
        }
        
        if (usesFeatureMapping) {
            // Handle mixed features
            if (!(input instanceof Map)) {
                throw new IllegalArgumentException("For mixed feature models, input must be Map<String, Object>");
            }
            
            @SuppressWarnings("unchecked")
            Map<String, Object> inputMap = (Map<String, Object>) input;
            return convertMixedFeatures(inputMap);
        } else {
            // Handle raw arrays
            if (!(input instanceof float[])) {
                throw new IllegalArgumentException("For simple models, input must be float[]");
            }
            return (float[]) input;
        }
    }
    
    private float[] convertMixedFeatures(Map<String, Object> input) {
        if (input.size() != featureNames.length) {
            throw new IllegalArgumentException(String.format(
                "Input must contain exactly %d features: %s. Got %d features: %s",
                featureNames.length, java.util.Arrays.toString(featureNames),
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
                    int index = dict.getIndex(value);
                    modelInput[i] = (float) index;
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
    
    private int getLabelIndex(String label) {
        return labelDictionary.getIndex(label);
    }
    
    private float[] createTargetVector(int classIndex) {
        // For multi-class, create one-hot vector
        // For binary, use single value
        Layer outputLayer = underlyingNet.getOutputLayer();
        
        if (outputLayer.getOutputSize() == 1) {
            // Binary classification
            return new float[]{(float) classIndex};
        } else {
            // Multi-class classification - one-hot encoding
            float[] target = new float[outputLayer.getOutputSize()];
            target[classIndex] = 1.0f;
            return target;
        }
    }
    
    
    private String[] generateFeatureNames(int count) {
        String[] names = new String[count];
        for (int i = 0; i < count; i++) {
            names[i] = "feature_" + i;
        }
        return names;
    }
    
    // ===============================
    // PUBLIC UTILITY METHODS
    // ===============================
    
    
    /**
     * Get all output class labels.
     * @return array of class labels in order
     */
    public String[] getOutputClasses() {
        // Get all values from the dictionary
        String[] classes = new String[labelDictionary.size()];
        for (int i = 0; i < labelDictionary.size(); i++) {
            Object value = labelDictionary.getValue(i);
            classes[i] = value != null ? (String) value : "class_" + i;
        }
        return classes;
    }
    
    /**
     * Get the number of output classes.
     * @return number of classes
     */
    public int getNumOutputClasses() {
        return labelDictionary.size();
    }
    
    /**
     * Set the sampling configuration for predictions.
     * 
     * <p><b>Examples:</b>
     * <pre>{@code
     * // For deterministic predictions (default)
     * classifier.setSamplingConfig(SamplingConfig.argmax());
     * 
     * // For more diverse predictions
     * classifier.setSamplingConfig(SamplingConfig.temperature(0.8f));
     * 
     * // For constrained vocabulary with diversity
     * classifier.setSamplingConfig(SamplingConfig.topK(10, 0.7f));
     * }</pre>
     * 
     * @param config the sampling configuration to use
     */
    public void setSamplingConfig(SamplingConfig config) {
        if (config == null) {
            throw new IllegalArgumentException("Sampling config cannot be null");
        }
        this.samplingConfig = config;
    }
    
    /**
     * Get the current sampling configuration.
     */
    public SamplingConfig getSamplingConfig() {
        return samplingConfig;
    }
    
    // ===============================
    // SERIALIZATION SUPPORT
    // ===============================
    
    /**
     * Save this SimpleNetString to a file.
     * 
     * @param path file path to save to
     * @throws IOException if save fails
     */
    public void save(Path path) throws IOException {
        try (DataOutputStream out = new DataOutputStream(java.nio.file.Files.newOutputStream(path))) {
            writeTo(out, SerializationConstants.CURRENT_VERSION);
        }
    }
    
    /**
     * Load a SimpleNetString from a file.
     * 
     * @param path file path to load from
     * @return loaded SimpleNetString
     * @throws IOException if load fails
     */
    public static SimpleNetString load(Path path) throws IOException {
        try (DataInputStream in = new DataInputStream(java.nio.file.Files.newInputStream(path))) {
            return deserialize(in, SerializationConstants.CURRENT_VERSION);
        }
    }
    
    @Override
    public void writeTo(DataOutputStream out, int version) throws IOException {
        // Write type identifier
        out.writeInt(getTypeId());
        
        // Write underlying neural network
        underlyingNet.writeTo(out, version);
        
        // Write feature configuration
        out.writeBoolean(usesFeatureMapping);
        if (usesFeatureMapping) {
            out.writeInt(featureNames.length);
            for (String featureName : featureNames) {
                out.writeUTF(featureName);
            }
            
            // Write feature dictionaries
            out.writeInt(featureDictionaries.size());
            for (Map.Entry<String, Dictionary> entry : featureDictionaries.entrySet()) {
                out.writeUTF(entry.getKey());
                entry.getValue().writeTo(out);
            }
        }
        
        // Write label dictionary
        labelDictionary.writeTo(out);
    }
    
    @Override
    public void readFrom(DataInputStream in, int version) throws IOException {
        throw new UnsupportedOperationException("Use deserialize(DataInputStream, int) static method instead");
    }
    
    /**
     * Deserialize a SimpleNetString from stream.
     */
    public static SimpleNetString deserialize(DataInputStream in, int version) throws IOException {
        // Read type identifier
        int typeId = in.readInt();
        if (typeId != SerializationConstants.TYPE_SIMPLE_NET_STRING) {
            String actualType = getTypeNameFromId(typeId);
            throw new IOException("Type mismatch: This file contains a " + actualType + 
                " model, but you're trying to load it as SimpleNetString. " +
                "Use " + actualType + ".load() instead.");
        }
        
        // Read underlying neural network
        NeuralNet underlyingNet = NeuralNet.deserialize(in, version);
        
        // Create SimpleNetString wrapper
        SimpleNetString simpleNet = new SimpleNetString(underlyingNet);
        
        // Read feature configuration
        boolean usesFeatureMapping = in.readBoolean();
        if (usesFeatureMapping) {
            int numFeatures = in.readInt();
            
            // Validate feature count matches
            if (numFeatures != simpleNet.featureNames.length) {
                throw new IOException(String.format(
                    "Feature count mismatch: expected %d, got %d", 
                    simpleNet.featureNames.length, numFeatures));
            }
            
            // Read feature names (but use generated ones for consistency)
            for (int i = 0; i < numFeatures; i++) {
                in.readUTF(); // Read but ignore saved feature names
            }
            
            // Read feature dictionaries
            int numDictionaries = in.readInt();
            for (int i = 0; i < numDictionaries; i++) {
                String featureName = in.readUTF();
                Dictionary dict = Dictionary.readFrom(in);
                simpleNet.featureDictionaries.put(featureName, dict);
            }
        }
        
        // Read label dictionary and replace the empty one
        Dictionary loadedLabelDict = Dictionary.readFrom(in);
        // Since labelDictionary is final, we need to copy its contents
        for (int i = 0; i < loadedLabelDict.size(); i++) {
            Object value = loadedLabelDict.getValue(i);
            if (value != null) {
                simpleNet.labelDictionary.getIndex(value);
            }
        }
        
        return simpleNet;
    }
    
    @Override
    public int getSerializedSize(int version) {
        int size = 4; // type ID
        size += underlyingNet.getSerializedSize(version);
        size += 1; // usesFeatureMapping
        
        if (usesFeatureMapping) {
            size += 4; // feature count
            for (String featureName : featureNames) {
                size += 2 + featureName.getBytes().length; // UTF string
            }
            
            size += 4; // dictionary count
            for (Map.Entry<String, Dictionary> entry : featureDictionaries.entrySet()) {
                size += 2 + entry.getKey().getBytes().length; // feature name
                size += entry.getValue().getSerializedSize();
            }
        }
        
        size += labelDictionary.getSerializedSize();
        return size;
    }
    
    @Override
    public int getTypeId() {
        return SerializationConstants.TYPE_SIMPLE_NET_STRING;
    }
    
    private static String getTypeNameFromId(int typeId) {
        switch (typeId) {
            case SerializationConstants.TYPE_SIMPLE_NET:
                return "SimpleNetInt";
            case SerializationConstants.TYPE_SIMPLE_NET_FLOAT:
                return "SimpleNetFloat";
            case SerializationConstants.TYPE_SIMPLE_NET_STRING:
                return "SimpleNetString";
            case SerializationConstants.TYPE_NEURAL_NET + 100:
                return "SimpleNetLanguageModel";
            default:
                return "Unknown type (ID: " + typeId + ")";
        }
    }
    
    // ===============================
    // SIMPLENET BASE CLASS METHODS
    // ===============================
    
    @Override
    protected void trainInternal(Object input, float[] targets) {
        float[] modelInput = convertInput(input);
        underlyingNet.train(modelInput, targets);
    }
    
    @Override
    protected float[] predictInternal(Object input) {
        float[] modelInput = convertInput(input);
        return underlyingNet.predict(modelInput);
    }
}