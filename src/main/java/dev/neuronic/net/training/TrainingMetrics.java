package dev.neuronic.net.training;

import java.time.Duration;
import java.time.Instant;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

/**
 * Comprehensive metrics collection during bulk training operations.
 * 
 * <p><b>Design Philosophy:</b>
 * <ul>
 *   <li><b>Non-intrusive:</b> Doesn't affect simple train() methods for online learning</li>
 *   <li><b>Comprehensive:</b> Tracks loss, accuracy, timing, and learning progress</li>
 *   <li><b>Thread-safe:</b> Can be used during parallel training operations</li>
 *   <li><b>Export-friendly:</b> Structured data suitable for JSON/CSV export</li>
 * </ul>
 * 
 * <p><b>Usage - Bulk Training with Metrics:</b>
 * <pre>{@code
 * // Create metrics tracker
 * TrainingMetrics metrics = new TrainingMetrics();
 * 
 * // Train with automatic metrics collection
 * SimpleNetInt classifier = SimpleNet.ofIntClassification(network);
 * classifier.trainBulk(trainingData, labels, metrics);
 * 
 * // Access training insights
 * System.out.printf("Final accuracy: %.2f%%\n", metrics.getFinalAccuracy() * 100);
 * System.out.printf("Training time: %s\n", metrics.getTotalTrainingTime());
 * System.out.printf("Best epoch: %d\n", metrics.getBestEpoch());
 * 
 * // Export for analysis
 * metrics.exportToJson("training_results.json");
 * }</pre>
 * 
 * <p><b>Thread Safety:</b> All methods are thread-safe for concurrent metric updates.
 */
public class TrainingMetrics {
    
    /**
     * Metrics for a single epoch of training.
     */
    public static class EpochMetrics {
        private final int epochNumber;
        private final double trainingLoss;
        private final double trainingAccuracy;
        private final double validationLoss;
        private final double validationAccuracy;
        private final Duration epochTime;
        private final int samplesSeen;
        
        public EpochMetrics(int epochNumber, double trainingLoss, double trainingAccuracy,
                           double validationLoss, double validationAccuracy, Duration epochTime,
                           int samplesSeen) {
            this.epochNumber = epochNumber;
            this.trainingLoss = trainingLoss;
            this.trainingAccuracy = trainingAccuracy;
            this.validationLoss = validationLoss;
            this.validationAccuracy = validationAccuracy;
            this.epochTime = epochTime;
            this.samplesSeen = samplesSeen;
        }
        
        // Getters
        public int getEpochNumber() { return epochNumber; }
        public double getTrainingLoss() { return trainingLoss; }
        public double getTrainingAccuracy() { return trainingAccuracy; }
        public double getValidationLoss() { return validationLoss; }
        public double getValidationAccuracy() { return validationAccuracy; }
        public Duration getEpochTime() { return epochTime; }
        public int getSamplesSeen() { return samplesSeen; }
        
        @Override
        public String toString() {
            return String.format("Epoch %d: train_loss=%.4f, train_acc=%.3f, val_loss=%.4f, val_acc=%.3f, time=%dms",
                epochNumber, trainingLoss, trainingAccuracy, validationLoss, validationAccuracy, epochTime.toMillis());
        }
    }
    
    /**
     * Configuration for training metrics collection.
     */
    public static class Config {
        private boolean trackValidation = true;
        private boolean trackTiming = true;
        private int validationFrequency = 1; // Every N epochs
        private boolean verbose = true;
        
        public Config trackValidation(boolean track) { this.trackValidation = track; return this; }
        public Config trackTiming(boolean track) { this.trackTiming = track; return this; }
        public Config validationFrequency(int frequency) { this.validationFrequency = frequency; return this; }
        public Config verbose(boolean verbose) { this.verbose = verbose; return this; }
        
        // Getters
        public boolean shouldTrackValidation() { return trackValidation; }
        public boolean shouldTrackTiming() { return trackTiming; }
        public int getValidationFrequency() { return validationFrequency; }
        public boolean isVerbose() { return verbose; }
    }
    
    private final Config config;
    private final List<EpochMetrics> epochHistory;
    private final Instant trainingStartTime;
    private volatile Instant trainingEndTime;
    private volatile int totalSamplesSeen;
    private volatile int bestEpoch;
    private volatile double bestValidationAccuracy;
    
    // Thread-safe metrics tracking
    private final Object metricsLock = new Object();
    
    /**
     * Create metrics tracker with default configuration.
     */
    public TrainingMetrics() {
        this(new Config());
    }
    
    /**
     * Create metrics tracker with custom configuration.
     */
    public TrainingMetrics(Config config) {
        this.config = config;
        this.epochHistory = Collections.synchronizedList(new ArrayList<>());
        this.trainingStartTime = Instant.now();
        this.bestEpoch = -1;
        this.bestValidationAccuracy = -1.0;
    }
    
    /**
     * Record metrics for a completed epoch.
     * Thread-safe for parallel training.
     */
    public void recordEpoch(int epochNumber, double trainingLoss, double trainingAccuracy,
                           double validationLoss, double validationAccuracy, 
                           int samplesSeen) {
        
        Duration epochTime = config.shouldTrackTiming() ? 
            Duration.between(getLastEpochEndTime(), Instant.now()) : Duration.ZERO;
        
        EpochMetrics metrics = new EpochMetrics(epochNumber, trainingLoss, trainingAccuracy,
                                              validationLoss, validationAccuracy, epochTime,
                                              samplesSeen);
        
        synchronized (metricsLock) {
            epochHistory.add(metrics);
            totalSamplesSeen += samplesSeen;
            
            // Track best validation performance
            if (validationAccuracy > bestValidationAccuracy) {
                bestValidationAccuracy = validationAccuracy;
                bestEpoch = epochNumber;
            }
        }
        
        if (config.isVerbose()) {
            System.out.println(metrics.toString());
        }
    }
    
    /**
     * Mark training as completed.
     */
    public void completeTraining() {
        this.trainingEndTime = Instant.now();
    }
    
    /**
     * Get total training duration.
     */
    public Duration getTotalTrainingTime() {
        Instant endTime = trainingEndTime != null ? trainingEndTime : Instant.now();
        return Duration.between(trainingStartTime, endTime);
    }
    
    /**
     * Get final training accuracy from last epoch.
     */
    public double getFinalAccuracy() {
        synchronized (metricsLock) {
            if (epochHistory.isEmpty()) return 0.0;
            return epochHistory.get(epochHistory.size() - 1).getTrainingAccuracy();
        }
    }
    
    /**
     * Get final validation accuracy from last epoch.
     */
    public double getFinalValidationAccuracy() {
        synchronized (metricsLock) {
            if (epochHistory.isEmpty()) return 0.0;
            return epochHistory.get(epochHistory.size() - 1).getValidationAccuracy();
        }
    }
    
    /**
     * Get epoch number with best validation accuracy.
     */
    public int getBestEpoch() {
        synchronized (metricsLock) {
            return bestEpoch;
        }
    }
    
    /**
     * Get best validation accuracy achieved.
     */
    public double getBestValidationAccuracy() {
        synchronized (metricsLock) {
            return bestValidationAccuracy;
        }
    }
    
    /**
     * Get total number of training samples seen.
     */
    public int getTotalSamplesSeen() {
        synchronized (metricsLock) {
            return totalSamplesSeen;
        }
    }
    
    /**
     * Get number of epochs completed.
     */
    public int getEpochCount() {
        synchronized (metricsLock) {
            return epochHistory.size();
        }
    }
    
    /**
     * Get metrics for a specific epoch.
     */
    public EpochMetrics getEpochMetrics(int epochNumber) {
        synchronized (metricsLock) {
            return epochHistory.stream()
                .filter(m -> m.getEpochNumber() == epochNumber)
                .findFirst()
                .orElse(null);
        }
    }
    
    /**
     * Get all epoch metrics (defensive copy).
     */
    public List<EpochMetrics> getAllEpochMetrics() {
        synchronized (metricsLock) {
            return new ArrayList<>(epochHistory);
        }
    }
    
    /**
     * Get training loss history as array.
     */
    public double[] getTrainingLossHistory() {
        synchronized (metricsLock) {
            return epochHistory.stream().mapToDouble(EpochMetrics::getTrainingLoss).toArray();
        }
    }
    
    /**
     * Get training accuracy history as array.
     */
    public double[] getTrainingAccuracyHistory() {
        synchronized (metricsLock) {
            return epochHistory.stream().mapToDouble(EpochMetrics::getTrainingAccuracy).toArray();
        }
    }
    
    /**
     * Get validation loss history as array.
     */
    public double[] getValidationLossHistory() {
        synchronized (metricsLock) {
            return epochHistory.stream().mapToDouble(EpochMetrics::getValidationLoss).toArray();
        }
    }
    
    /**
     * Get validation accuracy history as array.
     */
    public double[] getValidationAccuracyHistory() {
        synchronized (metricsLock) {
            return epochHistory.stream().mapToDouble(EpochMetrics::getValidationAccuracy).toArray();
        }
    }
    
    /**
     * Check if training appears to be overfitting.
     * Simple heuristic: validation accuracy declining while training accuracy improves.
     */
    public boolean isOverfitting() {
        synchronized (metricsLock) {
            if (epochHistory.size() < 3) return false;
            
            // Look at last 3 epochs
            int size = epochHistory.size();
            EpochMetrics recent = epochHistory.get(size - 1);
            EpochMetrics prev = epochHistory.get(size - 2);
            EpochMetrics older = epochHistory.get(size - 3);
            
            boolean trainingImproving = recent.getTrainingAccuracy() > prev.getTrainingAccuracy() &&
                                      prev.getTrainingAccuracy() > older.getTrainingAccuracy();
            boolean validationDeclining = recent.getValidationAccuracy() < prev.getValidationAccuracy() &&
                                        prev.getValidationAccuracy() < older.getValidationAccuracy();
            
            return trainingImproving && validationDeclining;
        }
    }
    
    /**
     * Get average samples processed per second.
     */
    public double getSamplesPerSecond() {
        synchronized (metricsLock) {
            if (totalSamplesSeen == 0) return 0.0;
            return totalSamplesSeen / (double) getTotalTrainingTime().toSeconds();
        }
    }
    
    /**
     * Get training summary for logging.
     */
    public String getSummary() {
        synchronized (metricsLock) {
            return String.format(
                "Training Summary:\n" +
                "  Epochs: %d\n" +
                "  Total samples: %,d\n" +
                "  Training time: %s\n" +
                "  Final accuracy: %.2f%%\n" +
                "  Best validation accuracy: %.2f%% (epoch %d)\n" +
                "  Samples/sec: %.0f\n" +
                "  Overfitting detected: %s",
                getEpochCount(),
                getTotalSamplesSeen(),
                formatDuration(getTotalTrainingTime()),
                getFinalAccuracy() * 100,
                getBestValidationAccuracy() * 100,
                getBestEpoch(),
                getSamplesPerSecond(),
                isOverfitting() ? "Yes" : "No"
            );
        }
    }
    
    private Instant getLastEpochEndTime() {
        synchronized (metricsLock) {
            if (epochHistory.isEmpty()) {
                return trainingStartTime;
            }
            // Approximate by adding up previous epoch times
            return trainingStartTime.plus(
                epochHistory.stream()
                    .map(EpochMetrics::getEpochTime)
                    .reduce(Duration.ZERO, Duration::plus)
            );
        }
    }
    
    private String formatDuration(Duration duration) {
        long seconds = duration.getSeconds();
        if (seconds < 60) {
            return String.format("%ds", seconds);
        } else if (seconds < 3600) {
            return String.format("%dm %ds", seconds / 60, seconds % 60);
        } else {
            long hours = seconds / 3600;
            long minutes = (seconds % 3600) / 60;
            return String.format("%dh %dm", hours, minutes);
        }
    }
}