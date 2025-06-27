package dev.neuronic.net.training;

import java.util.*;

/**
 * Automatic train/validation dataset splitting and evaluation utilities.
 * 
 * <p><b>Key Features:</b>
 * <ul>
 *   <li><b>Smart splitting:</b> Stratified splits for classification, random for regression</li>
 *   <li><b>Flexible ratios:</b> Configurable train/validation proportions</li>
 *   <li><b>Type-safe:</b> Generic support for different input/output types</li>
 *   <li><b>Reproducible:</b> Seed-based splitting for consistent results</li>
 *   <li><b>Efficient evaluation:</b> Batch processing for validation metrics</li>
 * </ul>
 * 
 * <p><b>Usage - Classification with Automatic Splitting:</b>
 * <pre>{@code
 * // Create evaluator with 80/20 train/validation split
 * ValidationEvaluator<float[], Integer> evaluator = ValidationEvaluator
 *     .forClassification(inputs, labels)
 *     .withSplit(0.8)
 *     .withSeed(42)
 *     .stratified(true);
 * 
 * // Get split datasets
 * TrainingData<float[], Integer> trainData = evaluator.getTrainingData();
 * TrainingData<float[], Integer> valData = evaluator.getValidationData();
 * 
 * // Evaluate model performance
 * ValidationResults results = evaluator.evaluate(classifier);
 * System.out.printf("Validation accuracy: %.2f%%\n", results.getAccuracy() * 100);
 * }</pre>
 * 
 * <p><b>Usage - Regression:</b>
 * <pre>{@code
 * ValidationEvaluator<float[], float[]> evaluator = ValidationEvaluator
 *     .forRegression(inputs, targets)
 *     .withSplit(0.8);
 * 
 * ValidationResults results = evaluator.evaluate(regressor);
 * System.out.printf("Validation MSE: %.4f\n", results.getMeanSquaredError());
 * }</pre>
 * 
 * <p><b>Thread Safety:</b> Evaluation methods are thread-safe for concurrent use.
 */
public class ValidationEvaluator<I, O> {
    
    /**
     * Container for training/validation data splits.
     */
    public static class TrainingData<I, O> {
        private final List<I> inputs;
        private final List<O> outputs;
        
        public TrainingData(List<I> inputs, List<O> outputs) {
            if (inputs.size() != outputs.size()) {
                throw new IllegalArgumentException("Input and output lists must have the same size");
            }
            this.inputs = new ArrayList<>(inputs);
            this.outputs = new ArrayList<>(outputs);
        }
        
        public List<I> getInputs() { return new ArrayList<>(inputs); }
        public List<O> getOutputs() { return new ArrayList<>(outputs); }
        public int size() { return inputs.size(); }
        
        public I getInput(int index) { return inputs.get(index); }
        public O getOutput(int index) { return outputs.get(index); }
        
        /**
         * Convert to arrays for batch processing.
         */
        @SuppressWarnings("unchecked")
        public I[] getInputArray(Class<I> inputClass) {
            return inputs.toArray((I[]) java.lang.reflect.Array.newInstance(inputClass, 0));
        }
        
        @SuppressWarnings("unchecked")
        public O[] getOutputArray(Class<O> outputClass) {
            return outputs.toArray((O[]) java.lang.reflect.Array.newInstance(outputClass, 0));
        }
    }
    
    /**
     * Validation evaluation results.
     */
    public static class ValidationResults {
        private final double accuracy;
        private final double loss;
        private final double meanSquaredError;
        private final double meanAbsoluteError;
        private final Map<String, Double> additionalMetrics;
        private final int totalSamples;
        private final long evaluationTimeMs;
        
        public ValidationResults(double accuracy, double loss, double mse, double mae,
                               Map<String, Double> additionalMetrics, int totalSamples, long evaluationTimeMs) {
            this.accuracy = accuracy;
            this.loss = loss;
            this.meanSquaredError = mse;
            this.meanAbsoluteError = mae;
            this.additionalMetrics = new HashMap<>(additionalMetrics);
            this.totalSamples = totalSamples;
            this.evaluationTimeMs = evaluationTimeMs;
        }
        
        public double getAccuracy() { return accuracy; }
        public double getLoss() { return loss; }
        public double getMeanSquaredError() { return meanSquaredError; }
        public double getMeanAbsoluteError() { return meanAbsoluteError; }
        public Map<String, Double> getAdditionalMetrics() { return new HashMap<>(additionalMetrics); }
        public int getTotalSamples() { return totalSamples; }
        public long getEvaluationTimeMs() { return evaluationTimeMs; }
        
        @Override
        public String toString() {
            return String.format("ValidationResults{accuracy=%.4f, loss=%.4f, mse=%.4f, mae=%.4f, samples=%d, time=%dms}",
                accuracy, loss, meanSquaredError, meanAbsoluteError, totalSamples, evaluationTimeMs);
        }
    }
    
    /**
     * Configuration for validation evaluation.
     */
    public static class Config {
        private double trainRatio = 0.8;
        private boolean stratified = false;
        private long seed = System.currentTimeMillis();
        private boolean shuffle = true;
        private int batchSize = 1000;
        
        public Config withSplit(double trainRatio) { this.trainRatio = trainRatio; return this; }
        public Config stratified(boolean stratified) { this.stratified = stratified; return this; }
        public Config withSeed(long seed) { this.seed = seed; return this; }
        public Config shuffle(boolean shuffle) { this.shuffle = shuffle; return this; }
        public Config batchSize(int batchSize) { this.batchSize = batchSize; return this; }
        
        // Getters
        public double getTrainRatio() { return trainRatio; }
        public boolean isStratified() { return stratified; }
        public long getSeed() { return seed; }
        public boolean shouldShuffle() { return shuffle; }
        public int getBatchSize() { return batchSize; }
    }
    
    private final TrainingData<I, O> trainingData;
    private final TrainingData<I, O> validationData;
    private final Config config;
    private final boolean isClassification;
    
    private ValidationEvaluator(List<I> inputs, List<O> outputs, Config config, boolean isClassification) {
        this.config = config;
        this.isClassification = isClassification;
        
        // Split data
        List<Integer> indices = createIndices(inputs.size());
        if (config.shouldShuffle()) {
            Collections.shuffle(indices, new Random(config.getSeed()));
        }
        
        int trainSize = (int) (inputs.size() * config.getTrainRatio());
        
        if (config.isStratified() && isClassification) {
            // Stratified split for classification
            Map<O, List<Integer>> labelIndices = groupByLabel(outputs, indices);
            List<Integer> trainIndices = new ArrayList<>();
            List<Integer> valIndices = new ArrayList<>();
            
            for (Map.Entry<O, List<Integer>> entry : labelIndices.entrySet()) {
                List<Integer> labelIdx = entry.getValue();
                int labelTrainSize = (int) (labelIdx.size() * config.getTrainRatio());
                
                trainIndices.addAll(labelIdx.subList(0, labelTrainSize));
                valIndices.addAll(labelIdx.subList(labelTrainSize, labelIdx.size()));
            }
            
            // Shuffle the final splits
            Collections.shuffle(trainIndices, new Random(config.getSeed()));
            Collections.shuffle(valIndices, new Random(config.getSeed() + 1));
            
            this.trainingData = createDataFromIndices(inputs, outputs, trainIndices);
            this.validationData = createDataFromIndices(inputs, outputs, valIndices);
        } else {
            // Simple random split
            List<Integer> trainIndices = indices.subList(0, trainSize);
            List<Integer> valIndices = indices.subList(trainSize, inputs.size());
            
            this.trainingData = createDataFromIndices(inputs, outputs, trainIndices);
            this.validationData = createDataFromIndices(inputs, outputs, valIndices);
        }
    }
    
    /**
     * Create evaluator for classification tasks.
     */
    public static <I, O> ValidationEvaluator<I, O> forClassification(List<I> inputs, List<O> outputs) {
        return new ValidationEvaluator<>(inputs, outputs, new Config().stratified(true), true);
    }
    
    /**
     * Create evaluator for regression tasks.
     */
    public static <I, O> ValidationEvaluator<I, O> forRegression(List<I> inputs, List<O> outputs) {
        return new ValidationEvaluator<>(inputs, outputs, new Config(), false);
    }
    
    /**
     * Create evaluator with custom configuration.
     */
    public static <I, O> ValidationEvaluator<I, O> withConfig(List<I> inputs, List<O> outputs, 
                                                             Config config, boolean isClassification) {
        return new ValidationEvaluator<>(inputs, outputs, config, isClassification);
    }
    
    /**
     * Update configuration and recreate splits.
     */
    public ValidationEvaluator<I, O> withSplit(double trainRatio) {
        Config newConfig = new Config()
            .withSplit(trainRatio)
            .stratified(config.isStratified())
            .withSeed(config.getSeed())
            .shuffle(config.shouldShuffle())
            .batchSize(config.getBatchSize());
        
        // Combine all data and re-split
        List<I> allInputs = new ArrayList<>();
        List<O> allOutputs = new ArrayList<>();
        allInputs.addAll(trainingData.getInputs());
        allInputs.addAll(validationData.getInputs());
        allOutputs.addAll(trainingData.getOutputs());
        allOutputs.addAll(validationData.getOutputs());
        
        return new ValidationEvaluator<>(allInputs, allOutputs, newConfig, isClassification);
    }
    
    /**
     * Update seed and recreate splits.
     */
    public ValidationEvaluator<I, O> withSeed(long seed) {
        Config newConfig = new Config()
            .withSplit(config.getTrainRatio())
            .stratified(config.isStratified())
            .withSeed(seed)
            .shuffle(config.shouldShuffle())
            .batchSize(config.getBatchSize());
        
        // Combine all data and re-split
        List<I> allInputs = new ArrayList<>();
        List<O> allOutputs = new ArrayList<>();
        allInputs.addAll(trainingData.getInputs());
        allInputs.addAll(validationData.getInputs());
        allOutputs.addAll(trainingData.getOutputs());
        allOutputs.addAll(validationData.getOutputs());
        
        return new ValidationEvaluator<>(allInputs, allOutputs, newConfig, isClassification);
    }
    
    /**
     * Enable/disable stratified splitting.
     */
    public ValidationEvaluator<I, O> stratified(boolean stratified) {
        if (!isClassification && stratified) {
            throw new IllegalArgumentException("Stratified splitting only supported for classification tasks");
        }
        
        Config newConfig = new Config()
            .withSplit(config.getTrainRatio())
            .stratified(stratified)
            .withSeed(config.getSeed())
            .shuffle(config.shouldShuffle())
            .batchSize(config.getBatchSize());
        
        // Combine all data and re-split
        List<I> allInputs = new ArrayList<>();
        List<O> allOutputs = new ArrayList<>();
        allInputs.addAll(trainingData.getInputs());
        allInputs.addAll(validationData.getInputs());
        allOutputs.addAll(trainingData.getOutputs());
        allOutputs.addAll(validationData.getOutputs());
        
        return new ValidationEvaluator<>(allInputs, allOutputs, newConfig, isClassification);
    }
    
    /**
     * Get training data split.
     */
    public TrainingData<I, O> getTrainingData() {
        return trainingData;
    }
    
    /**
     * Get validation data split.
     */
    public TrainingData<I, O> getValidationData() {
        return validationData;
    }
    
    /**
     * Evaluate a classification model on validation data.
     */
    public ValidationResults evaluateClassification(java.util.function.Function<I, O> predictor) {
        long startTime = System.currentTimeMillis();
        
        int correct = 0;
        double totalLoss = 0.0;
        List<I> valInputs = validationData.getInputs();
        List<O> valOutputs = validationData.getOutputs();
        
        for (int i = 0; i < valInputs.size(); i++) {
            I input = valInputs.get(i);
            O expectedOutput = valOutputs.get(i);
            O predictedOutput = predictor.apply(input);
            
            if (Objects.equals(expectedOutput, predictedOutput)) {
                correct++;
            }
            
            // Simple cross-entropy approximation for loss
            // Note: This is a simplified version - real loss would require raw network outputs
            totalLoss += Objects.equals(expectedOutput, predictedOutput) ? 0.0 : 1.0;
        }
        
        double accuracy = (double) correct / valInputs.size();
        double avgLoss = totalLoss / valInputs.size();
        long evaluationTime = System.currentTimeMillis() - startTime;
        
        return new ValidationResults(accuracy, avgLoss, 0.0, 0.0, 
                                   new HashMap<>(), valInputs.size(), evaluationTime);
    }
    
    /**
     * Evaluate a regression model on validation data.
     */
    public ValidationResults evaluateRegression(java.util.function.Function<I, float[]> predictor) {
        long startTime = System.currentTimeMillis();
        
        double totalMSE = 0.0;
        double totalMAE = 0.0;
        List<I> valInputs = validationData.getInputs();
        List<O> valOutputs = validationData.getOutputs();
        
        for (int i = 0; i < valInputs.size(); i++) {
            I input = valInputs.get(i);
            float[] expected = (float[]) valOutputs.get(i);
            float[] predicted = predictor.apply(input);
            
            if (expected.length != predicted.length) {
                throw new IllegalArgumentException("Expected and predicted output dimensions must match");
            }
            
            double mse = 0.0;
            double mae = 0.0;
            for (int j = 0; j < expected.length; j++) {
                double diff = expected[j] - predicted[j];
                mse += diff * diff;
                mae += Math.abs(diff);
            }
            
            totalMSE += mse / expected.length;
            totalMAE += mae / expected.length;
        }
        
        double avgMSE = totalMSE / valInputs.size();
        double avgMAE = totalMAE / valInputs.size();
        long evaluationTime = System.currentTimeMillis() - startTime;
        
        return new ValidationResults(0.0, avgMSE, avgMSE, avgMAE, 
                                   new HashMap<>(), valInputs.size(), evaluationTime);
    }
    
    /**
     * Get dataset statistics for analysis.
     */
    public Map<String, Object> getDatasetStatistics() {
        Map<String, Object> stats = new HashMap<>();
        stats.put("total_samples", trainingData.size() + validationData.size());
        stats.put("training_samples", trainingData.size());
        stats.put("validation_samples", validationData.size());
        stats.put("train_ratio", (double) trainingData.size() / (trainingData.size() + validationData.size()));
        stats.put("stratified", config.isStratified());
        stats.put("classification", isClassification);
        
        if (isClassification) {
            // Count class distribution
            Map<O, Integer> trainClassCounts = countClasses(trainingData.getOutputs());
            Map<O, Integer> valClassCounts = countClasses(validationData.getOutputs());
            stats.put("train_class_distribution", trainClassCounts);
            stats.put("validation_class_distribution", valClassCounts);
            stats.put("num_classes", trainClassCounts.size());
        }
        
        return stats;
    }
    
    // Helper methods
    
    private List<Integer> createIndices(int size) {
        List<Integer> indices = new ArrayList<>(size);
        for (int i = 0; i < size; i++) {
            indices.add(i);
        }
        return indices;
    }
    
    private Map<O, List<Integer>> groupByLabel(List<O> outputs, List<Integer> indices) {
        Map<O, List<Integer>> labelIndices = new HashMap<>();
        for (Integer idx : indices) {
            O label = outputs.get(idx);
            labelIndices.computeIfAbsent(label, k -> new ArrayList<>()).add(idx);
        }
        return labelIndices;
    }
    
    private TrainingData<I, O> createDataFromIndices(List<I> inputs, List<O> outputs, List<Integer> indices) {
        List<I> selectedInputs = new ArrayList<>();
        List<O> selectedOutputs = new ArrayList<>();
        
        for (Integer idx : indices) {
            selectedInputs.add(inputs.get(idx));
            selectedOutputs.add(outputs.get(idx));
        }
        
        return new TrainingData<>(selectedInputs, selectedOutputs);
    }
    
    private Map<O, Integer> countClasses(List<O> outputs) {
        Map<O, Integer> counts = new HashMap<>();
        for (O output : outputs) {
            counts.merge(output, 1, Integer::sum);
        }
        return counts;
    }
}