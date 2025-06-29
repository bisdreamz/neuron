package dev.neuronic.net.simple;

import dev.neuronic.net.NeuralNet;
import dev.neuronic.net.Dictionary;
import dev.neuronic.net.common.Utils;
import dev.neuronic.net.layers.Feature;
import dev.neuronic.net.layers.Layer;
import dev.neuronic.net.layers.MixedFeatureInputLayer;
import dev.neuronic.net.serialization.SerializationConstants;
import dev.neuronic.net.training.BatchTrainer;
import dev.neuronic.net.losses.CrossEntropyLoss;
import dev.neuronic.net.losses.Loss;
import dev.neuronic.net.training.MetricsLogger;
import dev.neuronic.net.training.TrainingMetrics;
import dev.neuronic.net.training.ValidationEvaluator;

import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.IOException;
import java.nio.file.Path;
import java.util.*;
import java.util.concurrent.ConcurrentHashMap;

/**
 * Type-safe neural network wrapper for integer classification (e.g., MNIST digit recognition).
 * 
 * <p><b>What it does:</b> Handles classification tasks where labels are integers and you want
 * a single integer result representing the predicted class.
 * 
 * <p><b>Perfect for:</b>
 * <ul>
 *   <li><b>MNIST:</b> Digit recognition (returns 0-9)</li>
 *   <li><b>CIFAR-10:</b> Object classification (returns 0-9)</li>
 *   <li><b>Custom datasets:</b> Where your classes are naturally numbered</li>
 * </ul>
 * 
 * <p><b>Key Benefits:</b>
 * <ul>
 *   <li><b>Type safety:</b> Returns primitive int - no casting needed</li>
 *   <li><b>Performance:</b> No boxing/unboxing overhead</li>
 *   <li><b>Automatic dictionary:</b> Builds label mapping during training</li>
 *   <li><b>Label preservation:</b> Returns original integer labels from training</li>
 * </ul>
 * 
 * <p><b>Example - MNIST:</b>
 * <pre>{@code
 * // Create neural network
 * NeuralNet net = NeuralNet.newBuilder()
 *     .input(784)  // 28x28 MNIST images
 *     .layer(Layers.hiddenDenseRelu(256))
 *     .layer(Layers.hiddenDenseRelu(64))
 *     .output(Layers.outputSoftmaxCrossEntropy(10));
 * 
 * // Create type-safe classifier
 * SimpleNetInt classifier = SimpleNet.ofIntClassification(net);
 * 
 * // Train with MNIST data - automatic label discovery
 * classifier.train(mnistPixels1, 7);  // First time seeing digit 7
 * classifier.train(mnistPixels2, 3);  // First time seeing digit 3
 * classifier.train(mnistPixels3, 7);  // Digit 7 again
 * 
 * // Predict - returns primitive int (no casting!)
 * int predictedDigit = classifier.predict(testPixels);
 * System.out.println("Predicted digit: " + predictedDigit);  // "Predicted digit: 7"
 * }</pre>
 * 
 * <p><b>Thread Safety:</b> All methods are thread-safe for concurrent training and prediction.
 */
public class SimpleNetInt extends SimpleNet<Integer> {
    
    private final Dictionary labelDictionary;
    
    // Auto-buffering for mini-batch training
    private volatile int autoBatchSize = 0;  // 0 = disabled
    private final List<Object> batchInputBuffer = Collections.synchronizedList(new ArrayList<>());
    private final List<Integer> batchLabelBuffer = Collections.synchronizedList(new ArrayList<>());
    
    /**
     * Create a SimpleNetInt without output names (for backward compatibility and deserialization).
     * Package-private.
     */
    SimpleNetInt(NeuralNet underlyingNet) {
        this(underlyingNet, null);
    }
    
    /**
     * Create a SimpleNetInt for integer classification.
     * Package-private - use SimpleNet.ofIntClassification() instead.
     */
    SimpleNetInt(NeuralNet underlyingNet, Set<String> outputNames) {
        super(underlyingNet, outputNames);
        this.labelDictionary = new Dictionary();
        // Feature mapping initialization is now handled in the base class
    }
    
    /**
     * Train the classifier with a single example.
     * 
     * <p><b>For MNIST-style raw arrays:</b>
     * <pre>{@code
     * classifier.train(mnistPixels, 7);  // 784-element float array + integer label
     * }</pre>
     * 
     * <p><b>For mixed features:</b>
     * <pre>{@code
     * classifier.train(Map.of(
     *     "feature1", "some_string",
     *     "feature2", 42.5f
     * ), 3);  // Mixed features + integer label
     * }</pre>
     * 
     * @param input either float[] for raw arrays or Map&lt;String, Object&gt; for mixed features
     * @param label integer class label (e.g., 0-9 for MNIST)
     */
    public void train(Object input, int label) {
        // If auto-batching is enabled, buffer the training sample
        if (autoBatchSize > 0) {
            synchronized (batchInputBuffer) {
                batchInputBuffer.add(input);
                batchLabelBuffer.add(label);
                
                // Train if batch is full
                if (batchInputBuffer.size() >= autoBatchSize) {
                    List<Object> inputs = new ArrayList<>(batchInputBuffer);
                    List<Integer> labels = new ArrayList<>(batchLabelBuffer);
                    batchInputBuffer.clear();
                    batchLabelBuffer.clear();
                    
                    // Train batch (outside synchronized block)
                    trainBatch(inputs, labels);
                }
            }
            return;
        }
        
        // Normal single-sample training
        float[] modelInput = convertAndGetModelInput(input);
        
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
     * @return predicted integer class label (same type as used in training)
     */
    public int predictInt(Object input) {
        float[] modelInput = convertAndGetModelInput(input);
        
        // Use the new predictArgmax method
        int predictedClassIndex = (int) underlyingNet.predictArgmax(modelInput);
        
        // Convert back to original label
        Object originalLabel = labelDictionary.getValue(predictedClassIndex);
        return originalLabel != null ? (Integer) originalLabel : predictedClassIndex;
    }
    
    /**
     * Get the top K predicted classes with their confidence scores.
     * 
     * @param input input data
     * @param k number of top predictions to return
     * @return array of top k class predictions, sorted by confidence (highest first)
     */
    public int[] predictTopK(Object input, int k) {
        float[] modelInput = convertAndGetModelInput(input);
        
        // Use the new predictTopK method
        float[] topKIndices = underlyingNet.predictTopK(modelInput, k);
        
        // Convert float indices to int and map to original labels
        int[] topKLabels = new int[topKIndices.length];
        for (int i = 0; i < topKIndices.length; i++) {
            int classIndex = (int) topKIndices[i];
            Object originalLabel = labelDictionary.getValue(classIndex);
            topKLabels[i] = originalLabel != null ? (Integer) originalLabel : classIndex;
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
        float[] modelInput = convertAndGetModelInput(input);
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
    public boolean hasSeenLabel(int label) {
        return labelDictionary.containsValue(label);
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
    public SimpleNetTrainingResult trainBulk(List<Object> trainInputs, List<Integer> trainLabels,
                                           List<Object> valInputs, List<Integer> valLabels,
                                           SimpleNetTrainingConfig config) {
        // Validate inputs
        if (trainInputs.size() != trainLabels.size())
            throw new IllegalArgumentException("Training inputs and labels must have the same size");
        
        if (valInputs != null && valLabels != null && valInputs.size() != valLabels.size())
            throw new IllegalArgumentException("Validation inputs and labels must have the same size");
        
        // Convert training data
        float[][] encodedTrainInputs = new float[trainInputs.size()][];
        float[][] encodedTrainTargets = new float[trainLabels.size()][];
        
        for (int i = 0; i < trainInputs.size(); i++) {
            encodedTrainInputs[i] = convertAndGetModelInput(trainInputs.get(i));
            encodedTrainTargets[i] = createTargetVector(getLabelIndex(trainLabels.get(i)));
        }
        
        // Convert validation data if provided
        float[][] encodedValInputs = null;
        float[][] encodedValTargets = null;
        
        if (valInputs != null && valLabels != null) {
            encodedValInputs = new float[valInputs.size()][];
            encodedValTargets = new float[valLabels.size()][];
            
            for (int i = 0; i < valInputs.size(); i++) {
                encodedValInputs[i] = convertAndGetModelInput(valInputs.get(i));
                encodedValTargets[i] = createTargetVector(getLabelIndex(valLabels.get(i)));
            }
        }
        
        // Use shared implementation with validation split
        return trainWithEncodedData(encodedTrainInputs, encodedTrainTargets, 
                                   encodedValInputs, encodedValTargets, config);
    }
    // trainBulk is now inherited from base class
    
    
    // Helper methods are now inherited from base class
    
    private int getLabelIndex(Integer label) {
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
    // AUTO-BATCHING CONFIGURATION
    // ===============================
    
    /**
     * Enable automatic mini-batch training.
     * When enabled, calls to train() will automatically buffer samples and
     * train in batches for better performance.
     * 
     * <p><b>Example - Enable auto-batching:</b>
     * <pre>{@code
     * // Create classifier with auto-batching
     * SimpleNetInt classifier = SimpleNet.ofIntClassification(net)
     *     .withAutoBatching(32);  // Automatically batch every 32 samples
     * 
     * // Now just call train() normally - batching happens automatically!
     * for (Event event : eventStream) {
     *     classifier.train(event.features, event.label);
     * }
     * 
     * // Flush any remaining samples when done
     * classifier.flushBatch();
     * }</pre>
     * 
     * @param batchSize size of mini-batches (recommended: 16-32)
     * @return this instance for method chaining
     */
    public SimpleNetInt withAutoBatching(int batchSize) {
        if (batchSize <= 0) {
            throw new IllegalArgumentException("Batch size must be positive");
        }
        this.autoBatchSize = batchSize;
        return this;
    }
    
    /**
     * Disable automatic batching (default).
     * 
     * @return this instance for method chaining
     */
    public SimpleNetInt withoutAutoBatching() {
        // Flush before disabling
        if (this.autoBatchSize > 0) {
            flushBatch();  // Train any remaining buffered samples
        }
        this.autoBatchSize = 0;
        return this;
    }
    
    /**
     * Get current auto-batch size (0 if disabled).
     */
    public int getAutoBatchSize() {
        return autoBatchSize;
    }
    
    /**
     * Check if auto-batching is enabled.
     */
    public boolean isAutoBatchingEnabled() {
        return autoBatchSize > 0;
    }
    
    /**
     * Flush any buffered training samples.
     * Call this when done training to ensure all samples are processed.
     * 
     * @return number of samples that were flushed
     */
    public int flushBatch() {
        if (autoBatchSize <= 0) {
            return 0;
        }
        
        synchronized (batchInputBuffer) {
            if (batchInputBuffer.isEmpty()) {
                return 0;
            }
            
            int count = batchInputBuffer.size();
            List<Object> inputs = new ArrayList<>(batchInputBuffer);
            List<Integer> labels = new ArrayList<>(batchLabelBuffer);
            batchInputBuffer.clear();
            batchLabelBuffer.clear();
            
            // Train the partial batch
            trainBatch(inputs, labels);
            return count;
        }
    }
    
    /**
     * Get number of samples currently buffered (waiting for batch to fill).
     */
    public int getBufferedSampleCount() {
        return batchInputBuffer.size();
    }
    
    // ===============================
    // MINI-BATCH TRAINING FOR ONLINE LEARNING
    // ===============================
    
    /**
     * Train the classifier with a mini-batch of examples.
     * 
     * <p><b>Benefits of mini-batch training:</b>
     * <ul>
     *   <li>More stable gradients (averaged over batch)</li>
     *   <li>Better hardware utilization (vectorization)</li>
     *   <li>Faster convergence than single-sample updates</li>
     *   <li>Recommended batch size: 16-32 for online learning</li>
     * </ul>
     * 
     * <p><b>Example - Online learning with buffering:</b>
     * <pre>{@code
     * List<Object> inputBuffer = new ArrayList<>();
     * List<Integer> labelBuffer = new ArrayList<>();
     * 
     * // Buffer incoming events
     * inputBuffer.add(newInput);
     * labelBuffer.add(newLabel);
     * 
     * // Train when buffer reaches desired size
     * if (inputBuffer.size() >= 32) {
     *     classifier.trainBatch(inputBuffer, labelBuffer);
     *     inputBuffer.clear();
     *     labelBuffer.clear();
     * }
     * }</pre>
     * 
     * @param inputs list of input data (same format as single train())
     * @param labels list of integer labels corresponding to inputs
     */
    public void trainBatch(java.util.List<Object> inputs, java.util.List<Integer> labels) {
        if (inputs.size() != labels.size()) {
            throw new IllegalArgumentException("Input and label lists must have the same size");
        }
        
        if (inputs.isEmpty()) {
            return; // Nothing to train
        }
        
        // Convert to arrays for batch training
        float[][] batchInputs = new float[inputs.size()][];
        float[][] batchTargets = new float[labels.size()][];
        
        for (int i = 0; i < inputs.size(); i++) {
            batchInputs[i] = convertAndGetModelInput(inputs.get(i));
            
            // Add label to dictionary if not seen before
            int classIndex = labelDictionary.getIndex(labels.get(i));
            batchTargets[i] = createTargetVector(classIndex);
        }
        
        // Use underlying network's batch training
        underlyingNet.trainBatch(batchInputs, batchTargets);
    }
    
    /**
     * Train the classifier with a mini-batch using arrays (more efficient).
     * 
     * @param inputs array of input data
     * @param labels array of integer labels
     */
    public void trainBatch(Object[] inputs, int[] labels) {
        if (inputs.length != labels.length) {
            throw new IllegalArgumentException("Input and label arrays must have the same length");
        }
        
        // Convert to lists and use the list-based method
        java.util.List<Object> inputList = java.util.Arrays.asList(inputs);
        java.util.List<Integer> labelList = new java.util.ArrayList<>(labels.length);
        for (int label : labels) {
            labelList.add(label);
        }
        
        trainBatch(inputList, labelList);
    }
    
    /**
     * Train the classifier with a mini-batch using Map-based inputs (type-safe).
     * 
     * <p><b>Example usage:</b>
     * <pre>{@code
     * List<Map<String, Object>> batchInputs = new ArrayList<>();
     * List<Integer> batchLabels = new ArrayList<>();
     * 
     * // Accumulate batch
     * for (Example ex : examples) {
     *     batchInputs.add(ex.getFeatures());
     *     batchLabels.add(ex.getLabel());
     * }
     * 
     * // Train as a batch
     * classifier.trainBatchMaps(batchInputs, batchLabels);
     * }</pre>
     * 
     * @param inputs list of Map inputs with feature names to values
     * @param labels list of integer labels corresponding to inputs
     */
    public void trainBatchMaps(java.util.List<Map<String, Object>> inputs, java.util.List<Integer> labels) {
        if (inputs.size() != labels.size()) {
            throw new IllegalArgumentException("Input and label lists must have the same size");
        }
        
        if (inputs.isEmpty()) {
            return; // Nothing to train
        }
        
        if (!usesFeatureMapping) {
            throw new IllegalArgumentException(
                "This model does not use mixed features. Use trainBatchArrays() instead.");
        }
        
        // Convert to arrays for batch training
        float[][] batchInputs = new float[inputs.size()][];
        float[][] batchTargets = new float[labels.size()][];
        
        for (int i = 0; i < inputs.size(); i++) {
            batchInputs[i] = convertFromMap(inputs.get(i));
            
            // Add label to dictionary if not seen before
            int classIndex = labelDictionary.getIndex(labels.get(i));
            batchTargets[i] = createTargetVector(classIndex);
        }
        
        // Use underlying network's batch training
        underlyingNet.trainBatch(batchInputs, batchTargets);
    }
    
    /**
     * Train the classifier with a mini-batch using float array inputs (type-safe).
     * 
     * <p><b>Example usage:</b>
     * <pre>{@code
     * List<float[]> batchInputs = new ArrayList<>();
     * List<Integer> batchLabels = new ArrayList<>();
     * 
     * // Accumulate batch
     * for (Example ex : examples) {
     *     batchInputs.add(ex.getPixelData());
     *     batchLabels.add(ex.getDigit());
     * }
     * 
     * // Train as a batch
     * classifier.trainBatchArrays(batchInputs, batchLabels);
     * }</pre>
     * 
     * @param inputs list of float array inputs
     * @param labels list of integer labels corresponding to inputs
     */
    public void trainBatchArrays(java.util.List<float[]> inputs, java.util.List<Integer> labels) {
        if (inputs.size() != labels.size()) {
            throw new IllegalArgumentException("Input and label lists must have the same size");
        }
        
        if (inputs.isEmpty()) {
            return; // Nothing to train
        }
        
        // Convert to arrays for batch training
        float[][] batchInputs = new float[inputs.size()][];
        float[][] batchTargets = new float[labels.size()][];
        
        for (int i = 0; i < inputs.size(); i++) {
            batchInputs[i] = convertFromFloatArray(inputs.get(i));
            
            // Add label to dictionary if not seen before
            int classIndex = labelDictionary.getIndex(labels.get(i));
            batchTargets[i] = createTargetVector(classIndex);
        }
        
        // Use underlying network's batch training
        underlyingNet.trainBatch(batchInputs, batchTargets);
    }
    
    /**
     * Predict classes for a batch of inputs.
     * 
     * @param inputs list of input data
     * @return list of predicted integer class labels
     */
    public java.util.List<Integer> predictBatch(java.util.List<Object> inputs) {
        if (inputs.isEmpty()) {
            return new java.util.ArrayList<>();
        }
        
        // Convert inputs
        float[][] batchInputs = new float[inputs.size()][];
        for (int i = 0; i < inputs.size(); i++) {
            batchInputs[i] = convertAndGetModelInput(inputs.get(i));
        }
        
        // Get batch predictions
        float[][] batchOutputs = underlyingNet.predictBatch(batchInputs);
        
        // Convert to class predictions
        java.util.List<Integer> predictions = new java.util.ArrayList<>(batchOutputs.length);
        for (float[] output : batchOutputs) {
            int predictedClassIndex = Utils.argmax(output);
            Object originalLabel = labelDictionary.getValue(predictedClassIndex);
            predictions.add(originalLabel != null ? (Integer) originalLabel : predictedClassIndex);
        }
        
        return predictions;
    }
    
    /**
     * Predict classes for a batch of inputs (array version).
     * 
     * @param inputs array of input data
     * @return array of predicted integer class labels
     */
    public int[] predictBatchInt(Object[] inputs) {
        java.util.List<Integer> predictions = predictBatch(java.util.Arrays.asList(inputs));
        return predictions.stream().mapToInt(Integer::intValue).toArray();
    }
    
    // ===============================
    // BULK TRAINING WITH METRICS
    // ===============================
    
    /**
     * Train on a dataset with automatic train/validation split and comprehensive metrics collection.
     * 
     * <p><b>Preserves online learning:</b> The simple train() method remains unchanged for online learning.
     * This bulk method is designed for batch training with performance tracking.</p>
     * 
     * <p><b>Example - MNIST with Metrics:</b>
     * <pre>{@code
     * // Prepare data
     * List<float[]> inputs = loadMnistImages();
     * List<Integer> labels = loadMnistLabels();
     * 
     * // Train with automatic metrics collection
     * TrainingMetrics metrics = new TrainingMetrics();
     * classifier.trainBulk(inputs, labels, 10, metrics);
     * 
     * // View results
     * System.out.printf("Final accuracy: %.2f%%\n", metrics.getFinalAccuracy() * 100);
     * MetricsLogger.printReport(metrics);
     * }</pre>
     * 
     * @param inputs list of input data (either float[] or Map<String, Object> for mixed features)
     * @param labels list of integer labels
     * @param epochs number of training epochs
     * @param metrics metrics collector for training progress
     * @return final validation accuracy
     */
    public double trainBulk(java.util.List<Object> inputs, java.util.List<Integer> labels, 
                           int epochs, TrainingMetrics metrics) {
        return trainBulk(inputs, labels, epochs, metrics, 
                        MetricsLogger.ProgressCallbacks.printEvery(1));
    }
    
    /**
     * Train on a dataset with custom progress callback.
     * 
     * @param inputs list of input data 
     * @param labels list of integer labels
     * @param epochs number of training epochs
     * @param metrics metrics collector for training progress
     * @param progressCallback callback for training progress updates (null for silent)
     * @return final validation accuracy
     */
    public double trainBulk(java.util.List<Object> inputs, java.util.List<Integer> labels, 
                           int epochs, TrainingMetrics metrics,
                           MetricsLogger.ProgressCallback progressCallback) {
        return trainBulk(inputs, labels, epochs, 0.8, metrics, progressCallback);
    }
    
    /**
     * Train on a dataset with custom train/validation split ratio.
     * 
     * @param inputs list of input data
     * @param labels list of integer labels
     * @param epochs number of training epochs
     * @param trainRatio ratio of data to use for training (rest for validation)
     * @param metrics metrics collector for training progress
     * @param progressCallback callback for training progress updates (null for silent)
     * @return final validation accuracy
     */
    public double trainBulk(java.util.List<Object> inputs, java.util.List<Integer> labels, 
                           int epochs, double trainRatio, TrainingMetrics metrics,
                           MetricsLogger.ProgressCallback progressCallback) {
        
        if (inputs.size() != labels.size()) {
            throw new IllegalArgumentException("Input and label lists must have the same size");
        }
        
        if (trainRatio <= 0.0 || trainRatio >= 1.0) {
            throw new IllegalArgumentException("Train ratio must be between 0 and 1");
        }
        
        // Create validation evaluator with stratified split
        ValidationEvaluator<Object, Integer> evaluator =
            ValidationEvaluator.forClassification(inputs, labels)
                .withSplit(trainRatio)
                .stratified(true);
        
        // Get split datasets
        ValidationEvaluator.TrainingData<Object, Integer> trainData =
            evaluator.getTrainingData();
        ValidationEvaluator.TrainingData<Object, Integer> valData =
            evaluator.getValidationData();
        
        System.out.printf("Training on %d samples, validating on %d samples\n", 
                         trainData.size(), valData.size());
        
        double bestValAccuracy = 0.0;
        
        // Training loop with metrics collection
        for (int epoch = 0; epoch < epochs; epoch++) {
            long epochStart = System.currentTimeMillis();
            
            // Training phase
            double trainingLoss = 0.0;
            int trainingCorrect = 0;
            
            java.util.List<Object> epochInputs = trainData.getInputs();
            java.util.List<Integer> epochLabels = trainData.getOutputs();
            
            // Shuffle training data each epoch
            java.util.List<Integer> indices = new java.util.ArrayList<>();
            for (int i = 0; i < epochInputs.size(); i++) {
                indices.add(i);
            }
            java.util.Collections.shuffle(indices);
            
            // Train on shuffled data
            for (int idx : indices) {
                Object input = epochInputs.get(idx);
                Integer label = epochLabels.get(idx);
                
                // Get prediction for accuracy calculation
                int prediction = predictInt(input);
                if (prediction == label) {
                    trainingCorrect++;
                }
                
                // Train the model (online learning preserved)
                train(input, label);
                
                // Simplified loss calculation (would need raw network output for true loss)
                trainingLoss += (prediction == label) ? 0.0 : 1.0;
            }
            
            double trainingAccuracy = (double) trainingCorrect / trainData.size();
            double avgTrainingLoss = trainingLoss / trainData.size();
            
            // Validation phase
            ValidationEvaluator.ValidationResults valResults =
                evaluator.evaluateClassification(this::predictInt);
            
            double validationAccuracy = valResults.getAccuracy();
            double validationLoss = valResults.getLoss();
            
            if (validationAccuracy > bestValAccuracy) {
                bestValAccuracy = validationAccuracy;
            }
            
            // Record epoch metrics
            long epochTime = System.currentTimeMillis() - epochStart;
            java.time.Duration epochDuration = java.time.Duration.ofMillis(epochTime);
            
            metrics.recordEpoch(epoch, avgTrainingLoss, trainingAccuracy, 
                              validationLoss, validationAccuracy, 
                              trainData.size());
            
            // Call progress callback if provided
            if (progressCallback != null) {
                TrainingMetrics.EpochMetrics epochMetrics =
                    metrics.getEpochMetrics(epoch);
                if (epochMetrics != null) {
                    progressCallback.onEpochComplete(epoch, epochMetrics);
                }
            }
        }
        
        // Complete training
        metrics.completeTraining();
        
        return bestValAccuracy;
    }
    
    /**
     * Evaluate model performance on a separate test dataset.
     * 
     * @param testInputs test input data
     * @param testLabels test labels
     * @return validation results with accuracy and other metrics
     */
    public ValidationEvaluator.ValidationResults evaluate(
            java.util.List<Object> testInputs, java.util.List<Integer> testLabels) {
        
        if (testInputs.size() != testLabels.size()) {
            throw new IllegalArgumentException("Test input and label lists must have the same size");
        }
        
        // Create evaluator just for evaluation (no split needed)
        ValidationEvaluator<Object, Integer> evaluator =
            ValidationEvaluator.forClassification(testInputs, testLabels);
        
        return evaluator.evaluateClassification(this::predictInt);
    }
    
    /**
     * Get current model performance metrics on given data.
     * 
     * @param inputs input data
     * @param labels true labels
     * @return map with accuracy, per-class accuracy, and other metrics
     */
    public java.util.Map<String, Double> getPerformanceMetrics(java.util.List<Object> inputs, 
                                                              java.util.List<Integer> labels) {
        if (inputs.size() != labels.size()) {
            throw new IllegalArgumentException("Input and label lists must have the same size");
        }
        
        java.util.Map<String, Double> metrics = new java.util.HashMap<>();
        
        int correct = 0;
        java.util.Map<Integer, Integer> classCorrect = new java.util.HashMap<>();
        java.util.Map<Integer, Integer> classTotal = new java.util.HashMap<>();
        
        for (int i = 0; i < inputs.size(); i++) {
            Object input = inputs.get(i);
            Integer trueLabel = labels.get(i);
            int prediction = predictInt(input);
            
            if (prediction == trueLabel) {
                correct++;
                classCorrect.merge(trueLabel, 1, Integer::sum);
            }
            classTotal.merge(trueLabel, 1, Integer::sum);
        }
        
        // Overall accuracy
        metrics.put("accuracy", (double) correct / inputs.size());
        
        // Per-class accuracy
        for (java.util.Map.Entry<Integer, Integer> entry : classTotal.entrySet()) {
            Integer classLabel = entry.getKey();
            Integer total = entry.getValue();
            Integer correctForClass = classCorrect.getOrDefault(classLabel, 0);
            double classAccuracy = (double) correctForClass / total;
            metrics.put("class_" + classLabel + "_accuracy", classAccuracy);
        }
        
        // Number of classes seen
        metrics.put("num_classes", (double) classTotal.size());
        metrics.put("total_samples", (double) inputs.size());
        
        return metrics;
    }
    
    // ===============================
    // PUBLIC UTILITY METHODS
    // ===============================
    
    /**
     * Get all output class labels.
     * @return array of class labels in order
     */
    public Integer[] getOutputClasses() {
        // Get all values from the dictionary
        Integer[] classes = new Integer[labelDictionary.size()];
        for (int i = 0; i < labelDictionary.size(); i++) {
            Object value = labelDictionary.getValue(i);
            classes[i] = value != null ? (Integer) value : i;
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
    
    // ===============================
    // SERIALIZATION SUPPORT
    // ===============================
    
    /**
     * Save this SimpleNetInt to a file.
     * 
     * @param path file path to save to
     * @throws IOException if save fails
     */
    public void save(Path path) throws IOException {
        writeTo(new DataOutputStream(java.nio.file.Files.newOutputStream(path)), SerializationConstants.CURRENT_VERSION);
    }
    
    /**
     * Load a SimpleNetInt from a file.
     * 
     * @param path file path to load from
     * @return loaded SimpleNetInt
     * @throws IOException if load fails
     */
    public static SimpleNetInt load(Path path) throws IOException {
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
     * Deserialize a SimpleNetInt from stream.
     */
    public static SimpleNetInt deserialize(DataInputStream in, int version) throws IOException {
        // Read type identifier
        int typeId = in.readInt();
        if (typeId != SerializationConstants.TYPE_SIMPLE_NET) {
            String actualType = getTypeNameFromId(typeId);
            throw new IOException("Type mismatch: This file contains a " + actualType + 
                " model, but you're trying to load it as SimpleNetInt. " +
                "Use " + actualType + ".load() instead.");
        }
        
        // Read underlying neural network
        NeuralNet underlyingNet = NeuralNet.deserialize(in, version);
        
        // Create SimpleNetInt wrapper
        SimpleNetInt simpleNet = new SimpleNetInt(underlyingNet);
        
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
            for (int i = 0; i < numFeatures; i++) {
                simpleNet.featureNames[i] = in.readUTF();
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
        return SerializationConstants.TYPE_SIMPLE_NET;
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
        return CrossEntropyLoss.INSTANCE;
    }
    
    @Override
    protected String getCheckpointMonitorMetric() {
        return "val_accuracy";  // For classification, monitor validation accuracy
    }
    
    @Override
    protected Object predictFromArray(float[] input) {
        float[] output = underlyingNet.predict(input);
        int predictedClass = Utils.argmax(output);
        
        // Return the original label (which might be different from the internal index)
        Object originalLabel = labelDictionary.getValue(predictedClass);
        return originalLabel != null ? (Integer) originalLabel : predictedClass;
    }
    
    @Override
    protected Object predictFromMap(Map<String, Object> input) {
        float[] modelInput = convertFromMap(input);
        float[] output = underlyingNet.predict(modelInput);
        int predictedClass = Utils.argmax(output);
        
        // Return the original label
        Object originalLabel = labelDictionary.getValue(predictedClass);
        return originalLabel != null ? (Integer) originalLabel : predictedClass;
    }
    
    @Override
    protected float[][] encodeTargets(List<Integer> targets) {
        float[][] encoded = new float[targets.size()][];
        
        for (int i = 0; i < targets.size(); i++) {
            int classIndex = getLabelIndex(targets.get(i));
            encoded[i] = createTargetVector(classIndex);
        }
        
        return encoded;
    }
}
