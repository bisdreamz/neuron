package dev.neuronic.net.simple;

import dev.neuronic.net.NeuralNet;
import dev.neuronic.net.Dictionary;
import dev.neuronic.net.layers.Feature;
import dev.neuronic.net.layers.Layer;
import dev.neuronic.net.layers.MixedFeatureInputLayer;
import dev.neuronic.net.losses.MseLoss;
import dev.neuronic.net.losses.Loss;
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
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.ConcurrentHashMap;

/**
 * Type-safe neural network wrapper for multi-output regression tasks (e.g., coordinate prediction, multi-metric scoring).
 * 
 * <p><b>What it does:</b> Handles regression tasks where you want to predict multiple numerical
 * values and get a float array result without casting.
 * 
 * <p><b>Perfect for:</b>
 * <ul>
 *   <li><b>Coordinate prediction:</b> (x, y) coordinates, 3D positions</li>
 *   <li><b>Multi-metric scoring:</b> Predict multiple scores simultaneously</li>
 *   <li><b>Portfolio optimization:</b> Multiple asset allocations</li>
 *   <li><b>Resource prediction:</b> Multiple resource requirements (CPU, memory, disk)</li>
 *   <li><b>Multi-dimensional output:</b> RGB color values, vector components</li>
 * </ul>
 * 
 * <p><b>Key Benefits:</b>
 * <ul>
 *   <li><b>Type safety:</b> Returns float[] - no casting needed</li>
 *   <li><b>Named outputs:</b> Optional output names for better interpretability</li>
 *   <li><b>Performance:</b> Direct array return eliminates object allocation</li>
 *   <li><b>Automatic features:</b> Handles mixed feature types automatically</li>
 *   <li><b>Batch processing:</b> Efficient multi-input prediction</li>
 * </ul>
 * 
 * <p><b>Example - Portfolio Allocation:</b>
 * <pre>{@code
 * // Create neural network for portfolio allocation (4 assets)
 * NeuralNet net = NeuralNet.newBuilder()
 *     .input(3)
 *     .layer(Layers.inputMixed(optimizer,
 *         Feature.embedding(100, 32),  // market_sector
 *         Feature.passthrough(),       // risk_tolerance  
 *         Feature.passthrough()        // investment_amount
 *     ))
 *     .layer(Layers.hiddenDenseRelu(128))
 *     .layer(Layers.hiddenDenseRelu(64))
 *     .output(Layers.outputLinearRegression(4));  // 4 asset allocations
 * 
 * // Create type-safe multi-output regressor with named outputs
 * String[] outputNames = {"stocks", "bonds", "real_estate", "cash"};
 * SimpleNetMultiFloat allocator = SimpleNet.ofMultiFloatRegression(net, outputNames);
 * 
 * // Train with portfolio data
 * allocator.train(Map.of(
 *     "market_sector", "technology",
 *     "risk_tolerance", 0.7f,
 *     "investment_amount", 100000.0f
 * ), new float[]{0.6f, 0.2f, 0.15f, 0.05f});  // Target allocations
 * 
 * // Predict - returns float[] directly (no casting!)
 * float[] allocation = allocator.predictMultiFloat(Map.of(
 *     "market_sector", "healthcare",
 *     "risk_tolerance", 0.4f,
 *     "investment_amount", 50000.0f
 * ));
 * 
 * System.out.printf("Allocation: stocks=%.2f, bonds=%.2f, real_estate=%.2f, cash=%.2f%n", 
 *                   allocation[0], allocation[1], allocation[2], allocation[3]);
 * }</pre>
 * 
 * <p><b>Thread Safety:</b> All methods are thread-safe for concurrent training and prediction.
 */
public class SimpleNetMultiFloat extends SimpleNet<float[]> {
    
    private final int outputCount;
    
    private boolean hasExplicitFeatureNames() {
        if (featureNames == null) return false;
        for (String name : featureNames) {
            if (name == null || name.startsWith("feature_")) return false;
        }
        return true;
    }
    
    /**
     * Create a SimpleNetMultiFloat without output names (for backward compatibility).
     * Package-private.
     */
    SimpleNetMultiFloat(NeuralNet underlyingNet) {
        this(underlyingNet, (Set<String>) null);
    }
    
    /**
     * Create a SimpleNetMultiFloat with array output names (for backward compatibility).
     * Package-private.
     */
    SimpleNetMultiFloat(NeuralNet underlyingNet, String[] outputNames) {
        this(underlyingNet, outputNames != null ? new java.util.LinkedHashSet<>(java.util.Arrays.asList(outputNames)) : null);
    }
    
    /**
     * Create a SimpleNetMultiFloat for multi-output regression.
     * Package-private - use SimpleNet.ofMultiFloatRegression() instead.
     */
    SimpleNetMultiFloat(NeuralNet underlyingNet, Set<String> outputNames) {
        super(underlyingNet, outputNames);
        this.outputCount = underlyingNet.getOutputLayer().getOutputSize();
    }
    
    /**
     * Train the regressor with a single example.
     * 
     * <p><b>For mixed features:</b>
     * <pre>{@code
     * regressor.train(Map.of(
     *     "feature1", "some_string",
     *     "feature2", 42.5f
     * ), new float[]{1.2f, 3.4f, 5.6f});  // Multiple targets
     * }</pre>
     * 
     * <p><b>For raw arrays:</b>
     * <pre>{@code
     * regressor.train(features, new float[]{1.2f, 3.4f, 5.6f});  // float[] + float[] targets
     * }</pre>
     * 
     * @param input either float[] for raw arrays or Map&lt;String, Object&gt; for mixed features
     * @param targets array of target values for each output
     */
    public void train(Object input, float[] targets) {
        if (targets.length != outputCount) {
            throw new IllegalArgumentException(String.format(
                "Target array length (%d) must match output count (%d)", 
                targets.length, outputCount));
        }
        
        float[] modelInput = convertAndGetModelInput(input);
        underlyingNet.train(modelInput, targets);
    }
    
    /**
     * Predict the values for new input.
     * 
     * @param input either float[] for raw arrays or Map&lt;String, Object&gt; for mixed features  
     * @return predicted float array with one value per output
     */
    public float[] predictMultiFloat(Object input) {
        float[] modelInput = convertAndGetModelInput(input);

        return underlyingNet.predict(modelInput);
    }
    
    
    /**
     * Make predictions on multiple inputs efficiently.
     * 
     * @param inputs array of input data
     * @return array of predicted float arrays in same order as inputs
     */
    public float[][] predict(Object[] inputs) {
        float[][] results = new float[inputs.length][];
        for (int i = 0; i < inputs.length; i++) {
            results[i] = predictMultiFloat(inputs[i]);
        }
        return results;
    }
    
    /**
     * Get the number of output values this regressor produces.
     */
    public int getOutputCount() {
        return outputCount;
    }
    
    /**
     * Check if this regressor has named outputs.
     */
    public boolean hasNamedOutputs() {
        return hasOutputNames();
    }
    
    /**
     * Get prediction statistics for debugging or confidence estimation.
     * 
     * @param input input data
     * @return the raw neural network output
     */
    public float[] getRawOutput(Object input) {
        float[] modelInput = convertAndGetModelInput(input);
        return underlyingNet.predict(modelInput);
    }
    
    /**
     * Get prediction variability estimate for each output by making multiple predictions.
     * This can be used as a rough confidence measure for multi-output regression.
     * 
     * @param input input data
     * @param samples number of prediction samples to take (default: 10)
     * @return array of standard deviations for each output (lower = more confident)
     */
    public float[] getPredictionVariability(Object input, int samples) {
        if (samples < 2) {
            throw new IllegalArgumentException("Need at least 2 samples to compute variability");
        }
        
        float[][] predictions = new float[samples][];
        for (int i = 0; i < samples; i++) {
            predictions[i] = predictMultiFloat(input);
        }
        
        // Compute standard deviation for each output
        float[] variabilities = new float[outputCount];
        for (int outputIdx = 0; outputIdx < outputCount; outputIdx++) {
            // Compute mean for this output
            float mean = 0;
            for (int sampleIdx = 0; sampleIdx < samples; sampleIdx++) {
                mean += predictions[sampleIdx][outputIdx];
            }
            mean /= samples;
            
            // Compute variance for this output
            float variance = 0;
            for (int sampleIdx = 0; sampleIdx < samples; sampleIdx++) {
                float diff = predictions[sampleIdx][outputIdx] - mean;
                variance += diff * diff;
            }
            variance /= (samples - 1);
            
            variabilities[outputIdx] = (float) Math.sqrt(variance);
        }
        
        return variabilities;
    }
    
    /**
     * Get prediction variability estimate with default 10 samples.
     * 
     * @param input input data
     * @return array of standard deviations for each output (lower = more confident)
     */
    public float[] getPredictionVariability(Object input) {
        return getPredictionVariability(input, 10);
    }
    
    /**
     * Get predictions with confidence intervals for each output.
     * 
     * @param input input data
     * @param confidenceLevel confidence level (e.g., 0.95 for 95% confidence)
     * @return 2D array where result[i] = [prediction, lowerBound, upperBound] for output i
     */
    public float[][] predictWithConfidence(Object input, float confidenceLevel) {
        if (confidenceLevel <= 0 || confidenceLevel >= 1) {
            throw new IllegalArgumentException("Confidence level must be between 0 and 1");
        }
        
        float[] predictions = predictMultiFloat(input);
        float[] variabilities = getPredictionVariability(input);
        
        // Use normal distribution approximation for confidence interval
        float z = getZScore(confidenceLevel);
        
        float[][] results = new float[outputCount][3];
        for (int i = 0; i < outputCount; i++) {
            float margin = z * variabilities[i];
            results[i][0] = predictions[i];           // prediction
            results[i][1] = predictions[i] - margin;  // lower bound
            results[i][2] = predictions[i] + margin;  // upper bound
        }
        
        return results;
    }
    
    /**
     * Get predictions with 95% confidence intervals for each output.
     * 
     * @param input input data
     * @return 2D array where result[i] = [prediction, lowerBound, upperBound] for output i
     */
    public float[][] predictWithConfidence(Object input) {
        return predictWithConfidence(input, 0.95f);
    }
    
    /**
     * Get named predictions with confidence intervals (only if output names were provided).
     * 
     * @param input input data
     * @param confidenceLevel confidence level (e.g., 0.95 for 95% confidence)
     * @return map from output names to confidence arrays [prediction, lowerBound, upperBound]
     * @throws UnsupportedOperationException if no output names were provided during construction
     */
    public Map<String, float[]> predictNamedWithConfidence(Object input, float confidenceLevel) {
        if (!hasOutputNames()) {
            throw new UnsupportedOperationException(
                "Named confidence prediction requires output names. Use SimpleNet.ofMultiFloatRegression(net, outputNames)");
        }
        
        float[][] confidenceResults = predictWithConfidence(input, confidenceLevel);
        Map<String, float[]> namedResults = new java.util.LinkedHashMap<>();
        int i = 0;
        for (String name : getOutputNames()) {
            namedResults.put(name, confidenceResults[i++]);
        }
        return namedResults;
    }
    
    /**
     * Get named predictions with 95% confidence intervals.
     * 
     * @param input input data
     * @return map from output names to confidence arrays [prediction, lowerBound, upperBound]
     * @throws UnsupportedOperationException if no output names were provided during construction
     */
    public Map<String, float[]> predictNamedWithConfidence(Object input) {
        return predictNamedWithConfidence(input, 0.95f);
    }
    
    private float getZScore(float confidenceLevel) {
        // Common confidence levels and their z-scores
        if (Math.abs(confidenceLevel - 0.90f) < 0.001f) return 1.645f;
        if (Math.abs(confidenceLevel - 0.95f) < 0.001f) return 1.96f;
        if (Math.abs(confidenceLevel - 0.99f) < 0.001f) return 2.576f;
        
        // For other levels, use approximation (good enough for most cases)
        float alpha = 1 - confidenceLevel;
        float halfAlpha = alpha / 2;
        
        if (halfAlpha <= 0.025f) return 1.96f;  // Default to 95% confidence
        if (halfAlpha <= 0.05f) return 1.645f;  // 90% confidence
        return 1.28f;  // 80% confidence
    }
    
    // Private helper methods
    
    // ===============================
    // BATCH TRAINING METHODS
    // ===============================
    
    /**
     * Train the network on a batch of samples using Map-based inputs.
     * This provides convenient batch training without requiring a full training configuration.
     * 
     * <p><b>Example usage:</b>
     * <pre>{@code
     * List<Map<String, Object>> batchInputs = new ArrayList<>();
     * List<float[]> batchTargets = new ArrayList<>();
     * 
     * // Accumulate batch (e.g., portfolio allocations)
     * for (Client client : clients) {
     *     batchInputs.add(client.getRiskProfile());
     *     batchTargets.add(new float[]{0.6f, 0.3f, 0.1f}); // stocks, bonds, cash
     * }
     * 
     * // Train as a batch
     * model.trainBatchMaps(batchInputs, batchTargets);
     * }</pre>
     * 
     * @param inputs list of Map inputs with feature names to values
     * @param targets list of target float arrays
     * @throws IllegalArgumentException if inputs and targets have different sizes
     */
    public void trainBatchMaps(List<Map<String, Object>> inputs, List<float[]> targets) {
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
     * <p><b>Example usage:</b>
     * <pre>{@code
     * List<float[]> batchInputs = new ArrayList<>();
     * List<float[]> batchTargets = new ArrayList<>();
     * 
     * // Accumulate batch
     * for (Example ex : examples) {
     *     batchInputs.add(ex.getInputFeatures());
     *     batchTargets.add(ex.getMultiOutputTargets());
     * }
     * 
     * // Train as a batch
     * model.trainBatchArrays(batchInputs, batchTargets);
     * }</pre>
     * 
     * @param inputs list of float array inputs
     * @param targets list of target float arrays
     * @throws IllegalArgumentException if inputs and targets have different sizes
     */
    public void trainBatchArrays(List<float[]> inputs, List<float[]> targets) {
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
    
    
    // ===============================
    // PUBLIC UTILITY METHODS
    // ===============================
    
    
    
    // ===============================
    // SERIALIZATION SUPPORT
    // ===============================
    
    /**
     * Save this SimpleNetMultiFloat to a file.
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
     * Load a SimpleNetMultiFloat from a file.
     * 
     * @param path file path to load from
     * @return loaded SimpleNetMultiFloat
     * @throws IOException if load fails
     */
    public static SimpleNetMultiFloat load(Path path) throws IOException {
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
        
        // Write output configuration
        out.writeInt(outputCount);
        boolean hasNames = hasOutputNames();
        out.writeBoolean(hasNames);
        if (hasNames) {
            Set<String> names = getOutputNames();
            out.writeInt(names.size());
            for (String outputName : names) {
                out.writeUTF(outputName);
            }
        }
        
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
    
    @Override
    public void readFrom(DataInputStream in, int version) throws IOException {
        throw new UnsupportedOperationException("Use deserialize(DataInputStream, int) static method instead");
    }
    
    /**
     * Deserialize a SimpleNetMultiFloat from stream.
     */
    public static SimpleNetMultiFloat deserialize(DataInputStream in, int version) throws IOException {
        // Read type identifier
        int typeId = in.readInt();
        if (typeId != SerializationConstants.TYPE_SIMPLE_NET_MULTI_FLOAT) {
            throw new IOException("Expected SimpleNetMultiFloat type ID, got: " + typeId);
        }
        
        // Read underlying neural network
        NeuralNet underlyingNet = NeuralNet.deserialize(in, version);
        
        // Read output configuration
        int outputCount = in.readInt();
        boolean hasOutputNames = in.readBoolean();
        String[] outputNames = null;
        if (hasOutputNames) {
            int nameCount = in.readInt();
            outputNames = new String[nameCount];
            for (int i = 0; i < nameCount; i++) {
                outputNames[i] = in.readUTF();
            }
        }
        
        // Create SimpleNetMultiFloat wrapper
        Set<String> outputNameSet = null;
        if (outputNames != null) {
            outputNameSet = new java.util.LinkedHashSet<>(java.util.Arrays.asList(outputNames));
        }
        SimpleNetMultiFloat simpleNet = new SimpleNetMultiFloat(underlyingNet, outputNameSet);
        
        // Validate output count matches
        if (outputCount != simpleNet.outputCount) {
            throw new IOException(String.format(
                "Output count mismatch: expected %d, got %d", 
                simpleNet.outputCount, outputCount));
        }
        
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
            
            // Read and restore feature names
            String[] savedFeatureNames = new String[numFeatures];
            for (int i = 0; i < numFeatures; i++) {
                savedFeatureNames[i] = in.readUTF();
            }
            // Replace generated names with saved names
            System.arraycopy(savedFeatureNames, 0, simpleNet.featureNames, 0, numFeatures);
            
            // Read feature dictionaries
            int numDictionaries = in.readInt();
            for (int i = 0; i < numDictionaries; i++) {
                String featureName = in.readUTF();
                
                // Find the feature index and get maxBounds
                int featureIndex = -1;
                for (int j = 0; j < simpleNet.featureNames.length; j++) {
                    if (simpleNet.featureNames[j].equals(featureName)) {
                        featureIndex = j;
                        break;
                    }
                }
                if (featureIndex == -1) {
                    throw new IOException("Unknown feature name in serialized data: " + featureName);
                }
                
                int maxBounds = simpleNet.features[featureIndex].getMaxUniqueValues();
                Dictionary dict = Dictionary.readFrom(in, maxBounds);
                simpleNet.featureDictionaries.put(featureName, dict);
            }
        }
        
        return simpleNet;
    }
    
    @Override
    public int getSerializedSize(int version) {
        int size = 4; // type ID
        size += underlyingNet.getSerializedSize(version);
        size += 4; // outputCount
        size += 1; // hasOutputNames
        
        if (hasOutputNames()) {
            size += 4; // output names count
            for (String outputName : getOutputNames()) {
                size += 2 + outputName.getBytes().length; // UTF string
            }
        }
        
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
    
    @Override
    public int getTypeId() {
        return SerializationConstants.TYPE_SIMPLE_NET_MULTI_FLOAT;
    }
    
    // ===============================
    // SIMPLENET BASE CLASS METHODS
    // ===============================
    
    @Override
    protected void trainInternal(Object input, float[] targets) {
        if (targets.length != outputCount) {
            throw new IllegalArgumentException(String.format(
                "Target array length (%d) must match output count (%d)", 
                targets.length, outputCount));
        }
        
        float[] modelInput = convertAndGetModelInput(input);
        underlyingNet.train(modelInput, targets);
    }
    
    @Override
    protected float[] predictInternal(Object input) {
        float[] modelInput = convertAndGetModelInput(input);
        return underlyingNet.predict(modelInput);
    }
    
    @Override
    public SimpleNetTrainingResult trainBulk(List<?> inputs, List<float[]> targets,
                                            SimpleNetTrainingConfig config) {
        if (inputs.size() != targets.size()) {
            throw new IllegalArgumentException("Inputs and targets must have the same size");
        }
        
        if (inputs.isEmpty()) {
            return trainWithEncodedData(new float[0][], new float[0][], config);
        }
        
        // Convert targets to array
        float[][] encodedTargets = targets.toArray(new float[0][]);
        
        // Validate target array lengths
        for (int i = 0; i < encodedTargets.length; i++) {
            if (encodedTargets[i].length != outputCount) {
                throw new IllegalArgumentException(String.format(
                    "Target array at index %d has length %d but %d outputs expected",
                    i, encodedTargets[i].length, outputCount));
            }
        }
        
        // Determine input type from first element
        Object firstInput = inputs.get(0);
        
        if (firstInput instanceof Map) {
            // Map-based inputs
            float[][] encodedInputs = new float[inputs.size()][];
            
            for (int i = 0; i < inputs.size(); i++) {
                @SuppressWarnings("unchecked")
                Map<String, Object> mapInput = (Map<String, Object>) inputs.get(i);
                encodedInputs[i] = convertFromMap(mapInput);
            }
            
            return trainWithEncodedData(encodedInputs, encodedTargets, config);
            
        } else if (firstInput instanceof float[]) {
            // Array-based inputs
            float[][] encodedInputs = new float[inputs.size()][];
            
            for (int i = 0; i < inputs.size(); i++) {
                encodedInputs[i] = (float[]) inputs.get(i);
            }
            
            return trainWithEncodedData(encodedInputs, encodedTargets, config);
            
        } else {
            throw new IllegalArgumentException(
                "Inputs must be either Map<String, Object> or float[]. Got: " + 
                firstInput.getClass().getSimpleName());
        }
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
        return underlyingNet.predict(input);
    }
    
    @Override
    protected Object predictFromMap(Map<String, Object> input) {
        float[] modelInput = convertFromMap(input);
        return underlyingNet.predict(modelInput);
    }
    
    @Override
    protected void validateMapInput() {
        if (!hasExplicitFeatureNames()) {
            throw new IllegalArgumentException(
                "Cannot use Map<String,Object> input without explicit feature names. " +
                "Either configure feature names when creating the layer (e.g., Feature.oneHot(4, \"connectionType\")) " +
                "or use float[] input instead.");
        }
    }
    
    @Override
    protected float[][] encodeTargets(List<float[]> targets) {
        // For multi-float regression, targets are already float arrays
        return targets.toArray(new float[0][]);
    }
}