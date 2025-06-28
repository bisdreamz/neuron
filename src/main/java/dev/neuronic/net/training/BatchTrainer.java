package dev.neuronic.net.training;

import dev.neuronic.net.NeuralNet;
import dev.neuronic.net.losses.Loss;
import dev.neuronic.net.layers.Layer;
import dev.neuronic.net.optimizers.Optimizer;

import java.io.IOException;
import java.util.*;
import java.util.concurrent.*;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicReference;

/**
 * Comprehensive batch training system with advanced features:
 * - True mini-batch gradient accumulation
 * - Progress callbacks with detailed metrics
 * - Automatic train/validation splits
 * - Early stopping with model checkpointing
 * - Learning rate scheduling
 * - Visualization support
 * - Foundation for future auto-tuning
 */
public class BatchTrainer {
    private final NeuralNet model;
    private final Loss loss;
    private final ExecutorService batchExecutor; // For parallel batch processing
    
    // Configuration
    private final TrainingConfig config;
    private final List<TrainingCallback> callbacks = new ArrayList<>();
    private final TrainingMetrics metrics = new TrainingMetrics();
    
    // State
    private final AtomicBoolean stopRequested = new AtomicBoolean(false);
    
    public static class TrainingConfig {
        public final int batchSize;
        public final int epochs;
        public final float validationSplit;
        public final boolean shuffle;
        public final long randomSeed;
        public final int verbosity; // 0=silent, 1=progress, 2=detailed
        public final int parallelBatches; // 0=auto-detect, >0=specific count
        public final LearningRateSchedule learningRateSchedule; // Optional learning rate schedule
        
        private TrainingConfig(Builder builder) {
            this.batchSize = builder.batchSize;
            this.epochs = builder.epochs;
            this.validationSplit = builder.validationSplit;
            this.shuffle = builder.shuffle;
            this.randomSeed = builder.randomSeed;
            this.verbosity = builder.verbosity;
            this.parallelBatches = builder.parallelBatches;
            this.learningRateSchedule = builder.learningRateSchedule;
        }
        
        public static class Builder {
            private int batchSize = 32;
            private int epochs = 100;
            private float validationSplit = 0.2f;
            private boolean shuffle = true;
            private long randomSeed = System.currentTimeMillis();
            private int verbosity = 1;
            private int parallelBatches = 0; // 0 = auto-detect
            private LearningRateSchedule learningRateSchedule = null; // Optional
            
            public Builder batchSize(int batchSize) {
                if (batchSize <= 0)
                    throw new IllegalArgumentException("Batch size must be positive");
                this.batchSize = batchSize;
                return this;
            }
            
            public Builder epochs(int epochs) {
                if (epochs <= 0)
                    throw new IllegalArgumentException("Epochs must be positive");
                this.epochs = epochs;
                return this;
            }
            
            public Builder validationSplit(float split) {
                if (split < 0 || split >= 1)
                    throw new IllegalArgumentException("Validation split must be in [0, 1)");
                this.validationSplit = split;
                return this;
            }
            
            public Builder shuffle(boolean shuffle) {
                this.shuffle = shuffle;
                return this;
            }
            
            public Builder randomSeed(long seed) {
                this.randomSeed = seed;
                return this;
            }
            
            public Builder verbosity(int level) {
                if (level < 0 || level > 2)
                    throw new IllegalArgumentException("Verbosity must be 0, 1, or 2");
                this.verbosity = level;
                return this;
            }
            
            public Builder parallelBatches(int count) {
                if (count < 0)
                    throw new IllegalArgumentException("parallelBatches must be >= 0");
                this.parallelBatches = count;
                return this;
            }
            
            public Builder withLearningRateSchedule(LearningRateSchedule schedule) {
                this.learningRateSchedule = schedule;
                return this;
            }
            
            public TrainingConfig build() {
                return new TrainingConfig(this);
            }
        }
    }
    
    public BatchTrainer(NeuralNet model, Loss loss, TrainingConfig config) {
        this.model = model;
        this.loss = loss;
        this.config = config;
        
        // Create batch executor based on parallelBatches config
        int parallelism = config.parallelBatches > 0 ? 
                         config.parallelBatches : 
                         detectOptimalParallelism(model);
        
        if (parallelism > 1)
            this.batchExecutor = Executors.newFixedThreadPool(parallelism);
        else
            this.batchExecutor = null; // Sequential processing
        
        // Don't setup default callbacks here - wait until fit() is called
    }
    
    /**
     * Detect optimal batch parallelism based on model complexity.
     * Small models: use all cores for batch parallelism
     * Large models: limit parallelism to avoid oversubscription
     */
    private int detectOptimalParallelism(NeuralNet model) {
        // TODO: Add getTotalParameters() to NeuralNet to calculate this properly
        // For now, use a simple heuristic based on layer count and sizes
        int cores = Runtime.getRuntime().availableProcessors();
        
        // Simple heuristic: if model has GRU/LSTM layers, limit parallelism
        // as these are already computationally intensive
        // Otherwise use all cores
        return cores; // For now, default to all cores
    }
    
    private void setupDefaultCallbacks() {
        // Progress reporting based on verbosity
        // Skip if a progress callback is already present (e.g., language model specific)
        if (config.verbosity > 0 && !hasProgressCallback()) {
            callbacks.add(new ProgressCallback(config.verbosity == 2, config.epochs));
        }
    }
    
    private boolean hasProgressCallback() {
        return callbacks.stream().anyMatch(cb -> cb instanceof ProgressCallback);
    }
    
    public BatchTrainer withCallback(TrainingCallback callback) {
        callbacks.add(callback);
        return this;
    }
    
    public BatchTrainer withEarlyStopping(int patience, float minDelta) {
        callbacks.add(new EarlyStoppingCallback(patience, minDelta, stopRequested));
        return this;
    }
    
    /**
     * Get the stop flag for external early stopping callbacks.
     * @return the AtomicBoolean stop flag
     */
    public AtomicBoolean getStopFlag() {
        return stopRequested;
    }
    
    public BatchTrainer withModelCheckpoint(String filepath, boolean saveOnlyBest) {
        callbacks.add(new ModelCheckpointCallback.WithModel(model, filepath, 
                                                           "val_accuracy", saveOnlyBest, 0));
        return this;
    }
    
    public BatchTrainer withLearningRateScheduler(LearningRateSchedulerCallback scheduler) {
        callbacks.add(scheduler);
        return this;
    }
    
    public BatchTrainer withVisualization(String outputDirectory) {
        callbacks.add(new VisualizationCallback(outputDirectory));
        return this;
    }
    
    /**
     * Main training method with automatic train/validation split
     */
    /**
     * Clean up resources. Should be called when training is complete.
     */
    public void close() {
        if (batchExecutor != null) {
            batchExecutor.shutdown();
            try {
                if (!batchExecutor.awaitTermination(60, TimeUnit.SECONDS))
                    batchExecutor.shutdownNow();
            } catch (InterruptedException e) {
                batchExecutor.shutdownNow();
                Thread.currentThread().interrupt();
            }
        }
    }
    
    public TrainingResult fit(float[][] inputs, float[][] targets) {
        // Split data
        DataSplit split = splitData(inputs, targets, config.validationSplit, config.randomSeed);
        
        // Train with split data
        return fit(split.trainX, split.trainY, split.valX, split.valY);
    }
    
    /**
     * Training method with pre-split data
     */
    public TrainingResult fit(float[][] trainX, float[][] trainY, 
                             float[][] valX, float[][] valY) {
        // Setup default callbacks if none have been added
        setupDefaultCallbacks();
        
        // Notify callbacks
        for (TrainingCallback callback : callbacks) {
            callback.onTrainingStart(model, metrics);
        }
        
        // Training loop
        for (int epoch = 0; epoch < config.epochs && !stopRequested.get(); epoch++) {
            long epochStart = System.currentTimeMillis();
            
            // Apply learning rate schedule if configured
            if (config.learningRateSchedule != null) {
                float newLearningRate = config.learningRateSchedule.getLearningRate(epoch, config.epochs);
                updateModelLearningRate(model, newLearningRate);
            }
            
            // Train one epoch
            EpochResult trainResult = trainEpoch(trainX, trainY, epoch);
            
            // Validation
            EpochResult valResult = null;
            if (valX != null && valX.length > 0) {
                if (config.verbosity > 0) {
                    System.out.println("\nValidating...");
                }
                valResult = validate(valX, valY);
            }
            
            // Record metrics
            long epochTime = System.currentTimeMillis() - epochStart;
            recordEpochMetrics(epoch, trainResult, valResult, epochTime);
            
            // Notify callbacks
            for (TrainingCallback callback : callbacks) {
                callback.onEpochEnd(epoch, metrics);
            }
        }
        
        // Debug: Log why training stopped
        if (stopRequested.get() && config.verbosity > 0) {
            System.out.println("\nTraining stopped early due to callback request.");
        }
        
        // Training complete
        metrics.completeTraining();
        
        for (TrainingCallback callback : callbacks) {
            callback.onTrainingEnd(model, metrics);
        }
        
        return new TrainingResult(metrics, this);
    }
    
    private EpochResult trainEpoch(float[][] inputs, float[][] targets, int epoch) {
        // Shuffle if needed
        int[] indices = new int[inputs.length];
        for (int i = 0; i < indices.length; i++) indices[i] = i;
        
        if (config.shuffle) {
            Random rng = new Random(config.randomSeed + epoch);
            for (int i = indices.length - 1; i > 0; i--) {
                int j = rng.nextInt(i + 1);
                int tmp = indices[i];
                indices[i] = indices[j];
                indices[j] = tmp;
            }
        }
        
        // Use parallel or sequential processing
        if (batchExecutor != null)
            return trainEpochParallel(inputs, targets, indices);
        else
            return trainEpochSequential(inputs, targets, indices);
    }
    
    private EpochResult trainEpochSequential(float[][] inputs, float[][] targets, int[] indices) {
        // Original sequential implementation
        float totalLoss = 0;
        int correctPredictions = 0;
        int totalSamples = 0;
        
        for (int i = 0; i < inputs.length; i += config.batchSize) {
            int batchEnd = Math.min(i + config.batchSize, inputs.length);
            int actualBatchSize = batchEnd - i;
            
            // Prepare batch
            float[][] batchX = new float[actualBatchSize][];
            float[][] batchY = new float[actualBatchSize][];
            
            for (int j = 0; j < actualBatchSize; j++) {
                int idx = indices[i + j];
                batchX[j] = inputs[idx];
                batchY[j] = targets[idx];
            }
            
            // Process batch
            BatchResult result = processBatch(batchX, batchY);
            totalLoss += result.totalLoss;
            correctPredictions += result.correctPredictions;
            totalSamples += actualBatchSize;
            
            // Progress callback for batch
            if (config.verbosity > 1) {
                int batchNum = i / config.batchSize + 1;
                int totalBatches = (inputs.length + config.batchSize - 1) / config.batchSize;
                for (TrainingCallback callback : callbacks) {
                    if (callback instanceof ProgressCallback) {
                        ((ProgressCallback) callback).onBatchEnd(batchNum, totalBatches, result);
                    }
                }
            }
        }
        
        float avgLoss = totalLoss / totalSamples;
        float accuracy = (float) correctPredictions / totalSamples;
        
        return new EpochResult(avgLoss, accuracy);
    }
    
    private EpochResult trainEpochParallel(float[][] inputs, float[][] targets, int[] indices) {
        int totalBatches = (inputs.length + config.batchSize - 1) / config.batchSize;
        int parallelism = batchExecutor instanceof ThreadPoolExecutor ? 
                         ((ThreadPoolExecutor) batchExecutor).getMaximumPoolSize() : 1;
        
        // Thread-safe accumulators for metrics
        AtomicReference<Float> totalLoss = new AtomicReference<>(0.0f);
        AtomicInteger correctPredictions = new AtomicInteger(0);
        AtomicInteger totalSamples = new AtomicInteger(0);
        AtomicInteger completedBatches = new AtomicInteger(0);
        
        // Use CompletionService for continuous batch processing
        CompletionService<BatchResult> completionService = new ExecutorCompletionService<>(batchExecutor);
        
        // Submit all batches immediately - executor will queue them
        for (int batchIndex = 0; batchIndex < totalBatches; batchIndex++) {
            final int batch = batchIndex;
            final int startIdx = batch * config.batchSize;
            final int endIdx = Math.min(startIdx + config.batchSize, inputs.length);
            
            completionService.submit(() -> {
                // Direct slice without copying - just create views
                int actualBatchSize = endIdx - startIdx;
                float[][] batchX = new float[actualBatchSize][];
                float[][] batchY = new float[actualBatchSize][];
                
                // Reference existing arrays instead of copying
                for (int j = 0; j < actualBatchSize; j++) {
                    int idx = indices[startIdx + j];
                    batchX[j] = inputs[idx];  // Reference, not copy
                    batchY[j] = targets[idx]; // Reference, not copy
                }
                
                return processBatch(batchX, batchY);
            });
        }
        
        // Collect results as they complete
        for (int i = 0; i < totalBatches; i++) {
            try {
                Future<BatchResult> future = completionService.take();
                BatchResult result = future.get();
                
                // Update metrics atomically
                totalLoss.updateAndGet(v -> v + result.totalLoss);
                correctPredictions.addAndGet(result.correctPredictions);
                totalSamples.addAndGet(result.batchSize);
                
                // Progress callback for each completed batch
                if (config.verbosity > 1) {
                    int batchNum = completedBatches.incrementAndGet();
                    for (TrainingCallback callback : callbacks) {
                        if (callback instanceof ProgressCallback) {
                            ((ProgressCallback) callback).onBatchEnd(batchNum, totalBatches, result);
                        }
                    }
                }
                
            } catch (Exception e) {
                throw new RuntimeException("Error in parallel batch training", e);
            }
        }
        
        float avgLoss = totalLoss.get() / totalSamples.get();
        float accuracy = (float) correctPredictions.get() / totalSamples.get();
        
        return new EpochResult(avgLoss, accuracy);
    }
    
    private BatchResult processBatch(float[][] batchX, float[][] batchY) {
        float totalLoss = 0;
        int correct = 0;
        
        // Get predictions before training for loss calculation
        float[][] predictions = model.predictBatch(batchX);
        
        // Calculate loss and accuracy
        for (int i = 0; i < batchX.length; i++) {
            totalLoss += loss.loss(predictions[i], batchY[i]);
            
            if (isClassification(batchY[i])) {
                int predicted = argmax(predictions[i]);
                int actual = argmax(batchY[i]);
                if (predicted == actual) correct++;
            }
        }
        
        // Train the batch - now fully thread-safe with lock-free forward/backward
        model.trainBatch(batchX, batchY);
        
        return new BatchResult(totalLoss, correct, batchX.length);
    }
    
    private EpochResult validate(float[][] valX, float[][] valY) {
        // Use parallel or sequential processing based on executor availability
        if (batchExecutor != null)
            return validateParallel(valX, valY);
        else
            return validateSequential(valX, valY);
    }
    
    private EpochResult validateSequential(float[][] valX, float[][] valY) {
        float totalLoss = 0;
        int correct = 0;
        
        // Process validation data in batches for efficiency
        for (int i = 0; i < valX.length; i += config.batchSize) {
            int batchEnd = Math.min(i + config.batchSize, valX.length);
            
            // Prepare batch
            float[][] batchX = Arrays.copyOfRange(valX, i, batchEnd);
            float[][] batchY = Arrays.copyOfRange(valY, i, batchEnd);
            
            // Get predictions
            float[][] predictions = model.predictBatch(batchX);
            
            // Calculate metrics
            for (int j = 0; j < predictions.length; j++) {
                totalLoss += loss.loss(predictions[j], batchY[j]);
                
                if (isClassification(batchY[j])) {
                    int predicted = argmax(predictions[j]);
                    int actual = argmax(batchY[j]);
                    if (predicted == actual) correct++;
                }
            }
        }
        
        float avgLoss = totalLoss / valX.length;
        float accuracy = (float) correct / valX.length;
        
        return new EpochResult(avgLoss, accuracy);
    }
    
    private EpochResult validateParallel(float[][] valX, float[][] valY) {
        int totalBatches = (valX.length + config.batchSize - 1) / config.batchSize;
        int parallelism = batchExecutor instanceof ThreadPoolExecutor ? 
                         ((ThreadPoolExecutor) batchExecutor).getMaximumPoolSize() : 1;
        
        // Thread-safe accumulators for metrics
        AtomicReference<Float> totalLoss = new AtomicReference<>(0.0f);
        AtomicInteger correctPredictions = new AtomicInteger(0);
        AtomicInteger totalSamples = new AtomicInteger(0);
        
        // Process batches in rounds of 'parallelism' concurrent batches
        for (int round = 0; round < totalBatches; round += parallelism) {
            int batchesInRound = Math.min(parallelism, totalBatches - round);
            List<Future<?>> futures = new ArrayList<>();
            
            for (int p = 0; p < batchesInRound; p++) {
                final int batchIndex = round + p;
                final int startIdx = batchIndex * config.batchSize;
                
                if (startIdx >= valX.length)
                    continue;
                
                Future<?> future = batchExecutor.submit(() -> {
                    try {
                        int batchEnd = Math.min(startIdx + config.batchSize, valX.length);
                        int actualBatchSize = batchEnd - startIdx;
                        
                        // Prepare batch
                        float[][] batchX = Arrays.copyOfRange(valX, startIdx, batchEnd);
                        float[][] batchY = Arrays.copyOfRange(valY, startIdx, batchEnd);
                        
                        // Get predictions (no training, just forward pass)
                        float[][] predictions = model.predictBatch(batchX);
                        
                        // Calculate metrics
                        float batchLoss = 0;
                        int batchCorrect = 0;
                        
                        for (int j = 0; j < predictions.length; j++) {
                            batchLoss += loss.loss(predictions[j], batchY[j]);
                            
                            if (isClassification(batchY[j])) {
                                int predicted = argmax(predictions[j]);
                                int actual = argmax(batchY[j]);
                                if (predicted == actual) batchCorrect++;
                            }
                        }
                        
                        // Update metrics (thread-safe)
                        final float finalBatchLoss = batchLoss;
                        totalLoss.updateAndGet(v -> v + finalBatchLoss);
                        correctPredictions.addAndGet(batchCorrect);
                        totalSamples.addAndGet(actualBatchSize);
                        
                        // Progress callback for validation batch (if verbosity > 1)
                        if (config.verbosity > 1) {
                            int completedBatches = totalSamples.get() / config.batchSize;
                            float currentAccuracy = totalSamples.get() > 0 ? 
                                (float) correctPredictions.get() / totalSamples.get() : 0;
                            System.out.printf("\rValidation batch %d/%d - acc: %.4f", 
                                completedBatches, totalBatches, currentAccuracy);
                        }
                        
                    } catch (Exception e) {
                        e.printStackTrace();
                        throw new RuntimeException("Error in validation batch processing", e);
                    }
                });
                futures.add(future);
            }
            
            // Wait for this round to complete before starting next
            for (Future<?> future : futures) {
                try {
                    future.get();
                } catch (Exception e) {
                    throw new RuntimeException("Error in parallel validation", e);
                }
            }
        }
        
        float avgLoss = totalLoss.get() / totalSamples.get();
        float accuracy = (float) correctPredictions.get() / totalSamples.get();
        
        // Clear progress line if we were showing it
        if (config.verbosity > 1)
            System.out.println();
        
        return new EpochResult(avgLoss, accuracy);
    }
    
    private void recordEpochMetrics(int epoch, EpochResult train, EpochResult val, long epochTime) {
        if (val != null) {
            metrics.recordEpoch(epoch, train.loss, train.accuracy, 
                               val.loss, val.accuracy, 
                               config.batchSize);
        } else {
            // No validation data
            metrics.recordEpoch(epoch, train.loss, train.accuracy, 
                               train.loss, train.accuracy, 
                               config.batchSize);
        }
    }
    
    private boolean isClassification(float[] target) {
        // Simple heuristic: if target is one-hot encoded, it's classification
        int ones = 0;
        for (float v : target) {
            if (v == 1.0f) ones++;
            else if (v != 0.0f) return false; // Not one-hot
        }
        return ones == 1;
    }
    
    private int argmax(float[] array) {
        int maxIdx = 0;
        float maxVal = array[0];
        for (int i = 1; i < array.length; i++) {
            if (array[i] > maxVal) {
                maxVal = array[i];
                maxIdx = i;
            }
        }
        return maxIdx;
    }
    
    private DataSplit splitData(float[][] inputs, float[][] targets, float valSplit, long seed) {
        int n = inputs.length;
        int valSize = (int) (n * valSplit);
        int trainSize = n - valSize;
        
        // Shuffle indices
        int[] indices = new int[n];
        for (int i = 0; i < n; i++) indices[i] = i;
        
        Random rng = new Random(seed);
        for (int i = n - 1; i > 0; i--) {
            int j = rng.nextInt(i + 1);
            int tmp = indices[i];
            indices[i] = indices[j];
            indices[j] = tmp;
        }
        
        // Split data
        float[][] trainX = new float[trainSize][];
        float[][] trainY = new float[trainSize][];
        float[][] valX = new float[valSize][];
        float[][] valY = new float[valSize][];
        
        for (int i = 0; i < trainSize; i++) {
            trainX[i] = inputs[indices[i]];
            trainY[i] = targets[indices[i]];
        }
        
        for (int i = 0; i < valSize; i++) {
            valX[i] = inputs[indices[trainSize + i]];
            valY[i] = targets[indices[trainSize + i]];
        }
        
        return new DataSplit(trainX, trainY, valX, valY);
    }
    
    public void stopTraining() {
        stopRequested.set(true);
    }
    
    private void updateModelLearningRate(NeuralNet model, float learningRate) {
        // Update learning rate for all optimizers in the model
        for (Layer layer : model.getLayers()) {
            Optimizer optimizer = layer.getOptimizer();
            if (optimizer != null) {
                optimizer.setLearningRate(learningRate);
            }
        }
    }
    
    // Helper classes
    
    private static class DataSplit {
        final float[][] trainX, trainY, valX, valY;
        
        DataSplit(float[][] trainX, float[][] trainY, float[][] valX, float[][] valY) {
            this.trainX = trainX;
            this.trainY = trainY;
            this.valX = valX;
            this.valY = valY;
        }
    }
    
    private static class EpochResult {
        final float loss;
        final float accuracy;
        
        EpochResult(float loss, float accuracy) {
            this.loss = loss;
            this.accuracy = accuracy;
        }
    }
    
    static class BatchResult {
        final float totalLoss;
        final int correctPredictions;
        final int batchSize;
        
        BatchResult(float totalLoss, int correctPredictions, int batchSize) {
            this.totalLoss = totalLoss;
            this.correctPredictions = correctPredictions;
            this.batchSize = batchSize;
        }
    }
    
    public static class TrainingResult implements AutoCloseable {
        private final TrainingMetrics metrics;
        private final BatchTrainer trainer;
        
        public TrainingResult(TrainingMetrics metrics, BatchTrainer trainer) {
            this.metrics = metrics;
            this.trainer = trainer;
        }
        
        public TrainingMetrics getMetrics() {
            return metrics;
        }
        
        public void exportMetrics(String filename) throws IOException {
            MetricsLogger.exportToJson(metrics, filename);
        }
        
        public void printSummary() {
            MetricsLogger.printReport(metrics);
        }
        
        @Override
        public void close() {
            trainer.close();
        }
    }
}