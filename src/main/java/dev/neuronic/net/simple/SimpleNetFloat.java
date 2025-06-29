package dev.neuronic.net.simple;

import dev.neuronic.net.NeuralNet;
import dev.neuronic.net.Dictionary;
import dev.neuronic.net.layers.Feature;
import dev.neuronic.net.layers.Layer;
import dev.neuronic.net.layers.MixedFeatureInputLayer;
import dev.neuronic.net.losses.MseLoss;
import dev.neuronic.net.losses.Loss;
import dev.neuronic.net.serialization.SerializationConstants;

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
public class SimpleNetFloat extends SimpleNet<Float> {
    
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
        // All feature mapping initialization is now handled in the base class
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
        float[] modelInput = convertAndGetModelInput(input);
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
        float[] modelInput = convertAndGetModelInput(input);
        
        // Get raw output
        float[] output = underlyingNet.predict(modelInput);
        
        // Return the single regression value
        return output[0];
    }
    
    /**
     * Predict using a float array (type-safe).
     * @param input raw float array
     * @return predicted float value
     */
    public float predictFloat(float[] input) {
        return (float) predict(input);
    }
    
    /**
     * Predict using a Map (type-safe).
     * @param input map of feature names to values
     * @return predicted float value
     */
    public float predictFloat(Map<String, Object> input) {
        return (float) predict(input);
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
        float[] modelInput = convertAndGetModelInput(input);
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
    
    // ===============================
    // BATCH TRAINING METHOD
    // ===============================
    
    /**
     * Train the network on a batch of samples using Map-based inputs.
     * This provides convenient batch training without requiring a full training configuration.
     * 
     * <p>Benefits over calling train() multiple times:
     * <ul>
     *   <li>Proper gradient accumulation across the batch</li>
     *   <li>Single weight update after processing all samples</li>
     *   <li>More stable learning and convergence</li>
     *   <li>Better performance (fewer weight updates)</li>
     * </ul>
     * 
     * <p>Example usage:
     * <pre>{@code
     * List<Map<String, Object>> batchInputs = new ArrayList<>();
     * List<Float> batchTargets = new ArrayList<>();
     * 
     * // Accumulate batch
     * for (Example ex : examples) {
     *     batchInputs.add(ex.getFeatures());
     *     batchTargets.add(ex.getTarget());
     * }
     * 
     * // Train as a batch
     * model.trainBatchMaps(batchInputs, batchTargets);
     * }</pre>
     * 
     * @param inputs list of Map inputs with feature names to values
     * @param targets list of target float values
     * @throws IllegalArgumentException if inputs and targets have different sizes
     */
    public void trainBatchMaps(List<Map<String, Object>> inputs, List<Float> targets) {
        if (inputs.size() != targets.size()) {
            throw new IllegalArgumentException(
                "Inputs and targets must have the same size. Got " + 
                inputs.size() + " inputs and " + targets.size() + " targets.");
        }
        
        if (inputs.isEmpty()) {
            return; // Nothing to train
        }
        
        if (!usesFeatureMapping) {
            throw new IllegalArgumentException(
                "This model does not use mixed features. Use trainBatchArrays() instead.");
        }
        
        // Convert inputs to float arrays
        float[][] encodedInputs = new float[inputs.size()][];
        for (int i = 0; i < inputs.size(); i++) {
            encodedInputs[i] = convertFromMap(inputs.get(i));
        }
        
        // Encode targets
        float[][] encodedTargets = encodeTargets(targets);
        
        // Use the underlying neural network's batch training
        underlyingNet.trainBatch(encodedInputs, encodedTargets);
    }
    
    /**
     * Train the network on a batch of samples using float array inputs.
     * This provides convenient batch training without requiring a full training configuration.
     * 
     * <p>Benefits over calling train() multiple times:
     * <ul>
     *   <li>Proper gradient accumulation across the batch</li>
     *   <li>Single weight update after processing all samples</li>
     *   <li>More stable learning and convergence</li>
     *   <li>Better performance (fewer weight updates)</li>
     * </ul>
     * 
     * <p>Example usage:
     * <pre>{@code
     * List<float[]> batchInputs = new ArrayList<>();
     * List<Float> batchTargets = new ArrayList<>();
     * 
     * // Accumulate batch
     * for (Example ex : examples) {
     *     batchInputs.add(ex.getFeatureArray());
     *     batchTargets.add(ex.getTarget());
     * }
     * 
     * // Train as a batch
     * model.trainBatchArrays(batchInputs, batchTargets);
     * }</pre>
     * 
     * @param inputs list of float array inputs
     * @param targets list of target float values
     * @throws IllegalArgumentException if inputs and targets have different sizes
     */
    public void trainBatchArrays(List<float[]> inputs, List<Float> targets) {
        if (inputs.size() != targets.size()) {
            throw new IllegalArgumentException(
                "Inputs and targets must have the same size. Got " + 
                inputs.size() + " inputs and " + targets.size() + " targets.");
        }
        
        if (inputs.isEmpty()) {
            return; // Nothing to train
        }
        
        // Convert inputs to float arrays
        float[][] encodedInputs = new float[inputs.size()][];
        for (int i = 0; i < inputs.size(); i++) {
            encodedInputs[i] = convertFromFloatArray(inputs.get(i));
        }
        
        // Encode targets
        float[][] encodedTargets = encodeTargets(targets);
        
        // Use the underlying neural network's batch training
        underlyingNet.trainBatch(encodedInputs, encodedTargets);
    }
    
    
    // trainBulk is now inherited from base class with common implementation
    
    // Helper methods are now inherited from base class
    
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
        float[] modelInput = convertAndGetModelInput(input);
        underlyingNet.train(modelInput, targets);
    }
    
    @Override
    protected float[] predictInternal(Object input) {
        float[] modelInput = convertAndGetModelInput(input);
        return underlyingNet.predict(modelInput);
    }
    
    @Override
    protected Loss getLossFunction() {
        return MseLoss.INSTANCE;
    }
    
    @Override
    protected String getCheckpointMonitorMetric() {
        return "val_loss";  // For regression, monitor validation loss
    }
    
    @Override
    protected Object predictFromArray(float[] input) {
        float[] output = underlyingNet.predict(input);
        return output[0];  // Return primitive float (auto-boxed)
    }
    
    @Override
    protected Object predictFromMap(Map<String, Object> input) {
        float[] modelInput = convertFromMap(input);
        float[] output = underlyingNet.predict(modelInput);
        return output[0];  // Return primitive float (auto-boxed)
    }
    
    @Override
    protected float[][] encodeTargets(List<Float> targets) {
        float[][] encoded = new float[targets.size()][1];
        for (int i = 0; i < targets.size(); i++) {
            encoded[i][0] = targets.get(i);
        }
        return encoded;
    }
}