package dev.neuronic.net.simple;

import dev.neuronic.net.NeuralNet;
import dev.neuronic.net.Dictionary;
import dev.neuronic.net.layers.Feature;
import dev.neuronic.net.layers.Layer;
import dev.neuronic.net.layers.MixedFeatureInputLayer;
import dev.neuronic.net.losses.MseLoss;
import dev.neuronic.net.serialization.SerializationConstants;
import dev.neuronic.net.training.BatchTrainer;
import dev.neuronic.net.training.TrainingCallback;
import dev.neuronic.net.training.EarlyStoppingCallback;
import dev.neuronic.net.training.ModelCheckpointCallback;
import dev.neuronic.net.training.VisualizationCallback;

import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.IOException;
import java.nio.file.Path;
import java.util.*;
import java.util.concurrent.ConcurrentHashMap;

/**
 * Type-safe neural network wrapper for single regression tasks (e.g., price prediction, scoring).
 * 
 * <p><b>What it does:</b> Handles regression tasks where you want to predict a single numerical
 * value and get a primitive float result without casting.
 * 
 * <p><b>Perfect for:</b>
 * <ul>
 *   <li><b>Price prediction:</b> House prices, stock prices, product costs</li>
 *   <li><b>Score prediction:</b> Credit scores, risk scores, quality ratings</li>
 *   <li><b>Physical measurements:</b> Temperature, weight, distance</li>
 *   <li><b>Performance metrics:</b> Speed, efficiency, accuracy scores</li>
 * </ul>
 * 
 * <p><b>Key Benefits:</b>
 * <ul>
 *   <li><b>Type safety:</b> Returns primitive float - no casting or boxing overhead</li>
 *   <li><b>Performance:</b> Direct float return eliminates object allocation</li>
 *   <li><b>Simple API:</b> Clean interface for single-value predictions</li>
 *   <li><b>Automatic features:</b> Handles mixed feature types automatically</li>
 * </ul>
 * 
 * <p><b>Example - House Price Prediction:</b>
 * <pre>{@code
 * // Create neural network for price prediction
 * NeuralNet net = NeuralNet.newBuilder()
 *     .input(4)
 *     .layer(Layers.inputMixed(optimizer,
 *         Feature.embedding(500, 32),   // neighborhood
 *         Feature.oneHot(5),            // house_type
 *         Feature.passthrough(),        // square_feet
 *         Feature.passthrough()         // bedrooms
 *     ))
 *     .layer(Layers.hiddenDenseRelu(128))
 *     .layer(Layers.hiddenDenseRelu(64))
 *     .output(Layers.outputLinearRegression(1));  // Single price output
 * 
 * // Create type-safe regressor
 * SimpleNetFloat pricer = SimpleNet.ofFloatRegression(net);
 * 
 * // Train with house data
 * pricer.train(Map.of(
 *     "neighborhood", "downtown",
 *     "house_type", 2,           // 2=condo
 *     "square_feet", 2400.0f,
 *     "bedrooms", 3.0f
 * ), 485000.0f);  // Target price
 * 
 * pricer.train(Map.of(
 *     "neighborhood", "suburbs",
 *     "house_type", 1,           // 1=single_family
 *     "square_feet", 3200.0f,
 *     "bedrooms", 4.0f
 * ), 625000.0f);
 * 
 * // Predict - returns primitive float (no casting!)
 * float predictedPrice = pricer.predictFloat(Map.of(
 *     "neighborhood", "downtown",
 *     "house_type", 1,
 *     "square_feet", 2800.0f,
 *     "bedrooms", 3.0f
 * ));
 * System.out.printf("Predicted price: $%.0f%n", predictedPrice);  // "Predicted price: $520000"
 * }</pre>
 * 
 * <p><b>Thread Safety:</b> All methods are thread-safe for concurrent training and prediction.
 */
public class SimpleNetFloat extends SimpleNet {
    
    private final boolean usesFeatureMapping;
    private final String[] featureNames;
    private final ConcurrentHashMap<String, Dictionary> featureDictionaries;
    private final Feature[] features;
    
    /**
     * Create a SimpleNetFloat without output names (for backward compatibility and deserialization).
     * Package-private.
     */
    SimpleNetFloat(NeuralNet underlyingNet) {
        this(underlyingNet, null);
    }
    
    /**
     * Create a SimpleNetFloat for single regression.
     * Package-private - use SimpleNet.ofFloatRegression() instead.
     */
    SimpleNetFloat(NeuralNet underlyingNet, Set<String> outputNames) {
        super(underlyingNet, outputNames);
        
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
     * Train the regressor with a single example.
     * 
     * <p><b>For mixed features:</b>
     * <pre>{@code
     * regressor.train(Map.of(
     *     "feature1", "some_string",
     *     "feature2", 42.5f
     * ), 125.75f);  // Mixed features + float target
     * }</pre>
     * 
     * <p><b>For raw arrays:</b>
     * <pre>{@code
     * regressor.train(features, 125.75f);  // float[] + float target
     * }</pre>
     * 
     * @param input either float[] for raw arrays or Map&lt;String, Object&gt; for mixed features
     * @param target numerical target value for regression
     */
    public void train(Object input, float target) {
        float[] modelInput = convertInput(input);
        float[] modelTarget = new float[]{target};
        
        underlyingNet.train(modelInput, modelTarget);
    }
    
    /**
     * Train the regressor with a single example (accepts any Number type).
     * 
     * @param input either float[] for raw arrays or Map&lt;String, Object&gt; for mixed features
     * @param target numerical target value (Float, Double, Integer, etc.)
     */
    public void train(Object input, Number target) {
        train(input, target.floatValue());
    }
    
    /**
     * Predict the value for new input.
     * 
     * @param input either float[] for raw arrays or Map&lt;String, Object&gt; for mixed features  
     * @return predicted float value
     */
    public float predictFloat(Object input) {
        float[] modelInput = convertInput(input);
        
        // Get raw output
        float[] output = underlyingNet.predict(modelInput);
        
        // Return the single regression value
        return output[0];
    }
    
    /**
     * Make predictions on multiple inputs efficiently.
     * 
     * @param inputs array of input data
     * @return array of predicted float values in same order as inputs
     */
    public float[] predict(Object[] inputs) {
        float[] results = new float[inputs.length];
        for (int i = 0; i < inputs.length; i++) {
            results[i] = predictFloat(inputs[i]);
        }
        return results;
    }
    
    /**
     * Get prediction statistics for debugging or confidence estimation.
     * 
     * @param input input data
     * @return the raw neural network output (usually just one value for regression)
     */
    public float[] getRawOutput(Object input) {
        float[] modelInput = convertInput(input);
        return underlyingNet.predict(modelInput);
    }
    
    /**
     * Get prediction variability estimate by making multiple predictions.
     * This can be used as a rough confidence measure for regression.
     * 
     * @param input input data
     * @param samples number of prediction samples to take (default: 10)
     * @return standard deviation of predictions (lower = more confident)
     */
    public float getPredictionVariability(Object input, int samples) {
        if (samples < 2) {
            throw new IllegalArgumentException("Need at least 2 samples to compute variability");
        }
        
        float[] predictions = new float[samples];
        for (int i = 0; i < samples; i++) {
            predictions[i] = predictFloat(input);
        }
        
        // Compute standard deviation
        float mean = 0;
        for (float pred : predictions) {
            mean += pred;
        }
        mean /= samples;
        
        float variance = 0;
        for (float pred : predictions) {
            float diff = pred - mean;
            variance += diff * diff;
        }
        variance /= (samples - 1);
        
        return (float) Math.sqrt(variance);
    }
    
    /**
     * Get prediction variability estimate with default 10 samples.
     * 
     * @param input input data
     * @return standard deviation of predictions (lower = more confident)
     */
    public float getPredictionVariability(Object input) {
        return getPredictionVariability(input, 10);
    }
    
    /**
     * Get prediction with confidence interval based on variability.
     * 
     * @param input input data
     * @param confidenceLevel confidence level (e.g., 0.95 for 95% confidence)
     * @return array with [prediction, lowerBound, upperBound]
     */
    public float[] predictWithConfidence(Object input, float confidenceLevel) {
        if (confidenceLevel <= 0 || confidenceLevel >= 1) {
            throw new IllegalArgumentException("Confidence level must be between 0 and 1");
        }
        
        float prediction = predictFloat(input);
        float variability = getPredictionVariability(input);
        
        // Use normal distribution approximation for confidence interval
        // For 95% confidence, z = 1.96; for 90%, z = 1.645; etc.
        float z = getZScore(confidenceLevel);
        float margin = z * variability;
        
        return new float[]{prediction, prediction - margin, prediction + margin};
    }
    
    /**
     * Get prediction with 95% confidence interval.
     * 
     * @param input input data
     * @return array with [prediction, lowerBound, upperBound]
     */
    public float[] predictWithConfidence(Object input) {
        return predictWithConfidence(input, 0.95f);
    }
    
    private float getZScore(float confidenceLevel) {
        // Common confidence levels and their z-scores
        if (Math.abs(confidenceLevel - 0.90f) < 0.001f) return 1.645f;
        if (Math.abs(confidenceLevel - 0.95f) < 0.001f) return 1.96f;
        if (Math.abs(confidenceLevel - 0.99f) < 0.001f) return 2.576f;
        
        // For other levels, use approximation (good enough for most cases)
        // This is a rough approximation of the inverse normal CDF
        float alpha = 1 - confidenceLevel;
        float halfAlpha = alpha / 2;
        
        if (halfAlpha <= 0.025f) return 1.96f;  // Default to 95% confidence
        if (halfAlpha <= 0.05f) return 1.645f;  // 90% confidence
        return 1.28f;  // 80% confidence
    }
    
    /**
     * Train the network with multiple examples using bulk training features.
     * Supports epochs, validation split, and callbacks.
     * 
     * @param inputs list of inputs (float[] or Map<String, Object>)
     * @param targets list of target values
     * @param config training configuration
     * @return training result with metrics
     */
    public SimpleNetTrainingResult trainBulk(List<Object> inputs, List<Float> targets, 
                                           SimpleNetTrainingConfig config) {
        if (inputs.size() != targets.size()) {
            throw new IllegalArgumentException("Inputs and targets must have the same size");
        }
        
        // Convert to arrays for BatchTrainer
        float[][] encodedInputs = new float[inputs.size()][];
        float[][] encodedTargets = new float[targets.size()][1];
        
        for (int i = 0; i < inputs.size(); i++) {
            encodedInputs[i] = convertInput(inputs.get(i));
            encodedTargets[i][0] = targets.get(i);
        }
        
        // Build callbacks
        List<TrainingCallback> callbacks = buildCallbacks(config);
        
        // Create BatchTrainer for regression (MSE loss)
        BatchTrainer trainer = new BatchTrainer(
            underlyingNet, 
            MseLoss.INSTANCE,
            config.getBatchConfig()
        );
        
        // Add callbacks
        for (TrainingCallback callback : callbacks)
            trainer.withCallback(callback);
        
        // Train
        long startTime = System.currentTimeMillis();
        BatchTrainer.TrainingResult batchResult = trainer.fit(encodedInputs, encodedTargets);
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
     * @param trainTargets training targets
     * @param valInputs validation inputs
     * @param valTargets validation targets
     * @param config training configuration
     * @return training result with metrics
     */
    public SimpleNetTrainingResult trainBulk(List<Object> trainInputs, List<Float> trainTargets,
                                           List<Object> valInputs, List<Float> valTargets,
                                           SimpleNetTrainingConfig config) {
        // Validate inputs
        if (trainInputs.size() != trainTargets.size())
            throw new IllegalArgumentException("Training inputs and targets must have the same size");
        
        if (valInputs != null && valTargets != null && valInputs.size() != valTargets.size())
            throw new IllegalArgumentException("Validation inputs and targets must have the same size");
        
        // Convert training data
        float[][] encodedTrainInputs = new float[trainInputs.size()][];
        float[][] encodedTrainTargets = new float[trainTargets.size()][1];
        
        for (int i = 0; i < trainInputs.size(); i++) {
            encodedTrainInputs[i] = convertInput(trainInputs.get(i));
            encodedTrainTargets[i][0] = trainTargets.get(i);
        }
        
        // Convert validation data if provided
        float[][] encodedValInputs = null;
        float[][] encodedValTargets = null;
        
        if (valInputs != null && valTargets != null) {
            encodedValInputs = new float[valInputs.size()][];
            encodedValTargets = new float[valTargets.size()][1];
            
            for (int i = 0; i < valInputs.size(); i++) {
                encodedValInputs[i] = convertInput(valInputs.get(i));
                encodedValTargets[i][0] = valTargets.get(i);
            }
        }
        
        // Build callbacks
        List<TrainingCallback> callbacks = buildCallbacks(config);
        
        // Create BatchTrainer for regression (MSE loss)
        BatchTrainer trainer = new BatchTrainer(
            underlyingNet, 
            MseLoss.INSTANCE,
            config.getBatchConfig()
        );
        
        // Add callbacks
        for (TrainingCallback callback : callbacks)
            trainer.withCallback(callback);
        
        // Train with pre-split data
        long startTime = System.currentTimeMillis();
        BatchTrainer.TrainingResult batchResult = trainer.fit(
            encodedTrainInputs, encodedTrainTargets, 
            encodedValInputs, encodedValTargets
        );
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
    public SimpleNetTrainingResult trainBulk(Object[] inputs, Float[] targets,
                                           SimpleNetTrainingConfig config) {
        return trainBulk(Arrays.asList(inputs), Arrays.asList(targets), config);
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
                "val_loss",  // For regression, monitor validation loss
                config.isCheckpointOnlyBest(),
                0
            ));
        }
        
        if (config.isVisualizationEnabled()) {
            callbacks.add(new VisualizationCallback(config.getVisualizationPath()));
        }
        
        return callbacks;
    }
    
    // Private helper methods
    
    public float[] convertInput(Object input) {
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
    
    // ===============================
    // SERIALIZATION SUPPORT
    // ===============================
    
    /**
     * Save this SimpleNetFloat to a file.
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
     * Load a SimpleNetFloat from a file.
     * 
     * @param path file path to load from
     * @return loaded SimpleNetFloat
     * @throws IOException if load fails
     */
    public static SimpleNetFloat load(Path path) throws IOException {
        try (DataInputStream in = new DataInputStream(java.nio.file.Files.newInputStream(path))) {
            return deserialize(in, SerializationConstants.CURRENT_VERSION);
        }
    }
    
    
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
    }
    
    
    public void readFrom(DataInputStream in, int version) throws IOException {
        throw new UnsupportedOperationException("Use deserialize(DataInputStream, int) static method instead");
    }
    
    /**
     * Deserialize a SimpleNetFloat from stream.
     */
    public static SimpleNetFloat deserialize(DataInputStream in, int version) throws IOException {
        // Read type identifier
        int typeId = in.readInt();
        if (typeId != SerializationConstants.TYPE_SIMPLE_NET_FLOAT) {
            String actualType = getTypeNameFromId(typeId);
            throw new IOException("Type mismatch: This file contains a " + actualType + 
                " model, but you're trying to load it as SimpleNetFloat. " +
                "Use " + actualType + ".load() instead.");
        }
        
        // Read underlying neural network
        NeuralNet underlyingNet = NeuralNet.deserialize(in, version);
        
        // Create SimpleNetFloat wrapper
        SimpleNetFloat simpleNet = new SimpleNetFloat(underlyingNet);
        
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
        
        return simpleNet;
    }
    
    
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
        
        return size;
    }
    
    
    public int getTypeId() {
        return SerializationConstants.TYPE_SIMPLE_NET_FLOAT;
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