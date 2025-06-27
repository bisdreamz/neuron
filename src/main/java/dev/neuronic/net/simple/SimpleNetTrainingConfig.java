package dev.neuronic.net.simple;

import dev.neuronic.net.training.BatchTrainer;
import dev.neuronic.net.training.LearningRateSchedule;

/**
 * Configuration for advanced training with SimpleNet.
 * Provides a user-friendly wrapper around BatchTrainer configuration
 * with sensible defaults for SimpleNet use cases.
 */
public class SimpleNetTrainingConfig {
    
    private final BatchTrainer.TrainingConfig batchConfig;
    private final boolean enableEarlyStopping;
    private final int earlyStoppingPatience;
    private final float earlyStoppingMinDelta;
    private final boolean enableCheckpointing;
    private final String checkpointPath;
    private final boolean checkpointOnlyBest;
    private final boolean enableVisualization;
    private final String visualizationPath;
    
    private SimpleNetTrainingConfig(Builder builder) {
        this.batchConfig = builder.batchConfigBuilder.build();
        this.enableEarlyStopping = builder.enableEarlyStopping;
        this.earlyStoppingPatience = builder.earlyStoppingPatience;
        this.earlyStoppingMinDelta = builder.earlyStoppingMinDelta;
        this.enableCheckpointing = builder.enableCheckpointing;
        this.checkpointPath = builder.checkpointPath;
        this.checkpointOnlyBest = builder.checkpointOnlyBest;
        this.enableVisualization = builder.enableVisualization;
        this.visualizationPath = builder.visualizationPath;
    }
    
    public BatchTrainer.TrainingConfig getBatchConfig() {
        return batchConfig;
    }
    
    public boolean isEarlyStoppingEnabled() {
        return enableEarlyStopping;
    }
    
    public int getEarlyStoppingPatience() {
        return earlyStoppingPatience;
    }
    
    public float getEarlyStoppingMinDelta() {
        return earlyStoppingMinDelta;
    }
    
    public boolean isCheckpointingEnabled() {
        return enableCheckpointing;
    }
    
    public String getCheckpointPath() {
        return checkpointPath;
    }
    
    public boolean isCheckpointOnlyBest() {
        return checkpointOnlyBest;
    }
    
    public boolean isVisualizationEnabled() {
        return enableVisualization;
    }
    
    public String getVisualizationPath() {
        return visualizationPath;
    }
    
    /**
     * Create a new builder with default settings optimized for SimpleNet.
     */
    public static Builder builder() {
        return new Builder();
    }
    
    public static class Builder {
        private final BatchTrainer.TrainingConfig.Builder batchConfigBuilder;
        private boolean enableEarlyStopping = false;
        private int earlyStoppingPatience = 10;
        private float earlyStoppingMinDelta = 0.001f;
        private boolean enableCheckpointing = false;
        private String checkpointPath = "models/checkpoint_{epoch}.nn";
        private boolean checkpointOnlyBest = true;
        private boolean enableVisualization = false;
        private String visualizationPath = "plots";
        
        private Builder() {
            this.batchConfigBuilder = new BatchTrainer.TrainingConfig.Builder()
                .batchSize(32)
                .epochs(100)
                .validationSplit(0.2f)
                .shuffle(true)
                .verbosity(1) // Default to progress bar
                .parallelBatches(0); // Auto-detect optimal parallelism
        }
        
        /**
         * Set batch size (default: 32).
         */
        public Builder batchSize(int batchSize) {
            batchConfigBuilder.batchSize(batchSize);
            return this;
        }
        
        /**
         * Set number of epochs (default: 100).
         */
        public Builder epochs(int epochs) {
            batchConfigBuilder.epochs(epochs);
            return this;
        }
        
        /**
         * Set validation split ratio (default: 0.2).
         */
        public Builder validationSplit(float split) {
            batchConfigBuilder.validationSplit(split);
            return this;
        }
        
        /**
         * Enable/disable shuffling (default: true).
         */
        public Builder shuffle(boolean shuffle) {
            batchConfigBuilder.shuffle(shuffle);
            return this;
        }
        
        /**
         * Set random seed for reproducibility.
         */
        public Builder randomSeed(long seed) {
            batchConfigBuilder.randomSeed(seed);
            return this;
        }
        
        /**
         * Set verbosity level: 0=silent, 1=progress, 2=detailed (default: 1).
         */
        public Builder verbosity(int level) {
            batchConfigBuilder.verbosity(level);
            return this;
        }
        
        /**
         * Enable early stopping with specified patience.
         */
        public Builder withEarlyStopping(int patience) {
            this.enableEarlyStopping = true;
            this.earlyStoppingPatience = patience;
            return this;
        }
        
        /**
         * Enable early stopping with patience and minimum delta.
         */
        public Builder withEarlyStopping(int patience, float minDelta) {
            this.enableEarlyStopping = true;
            this.earlyStoppingPatience = patience;
            this.earlyStoppingMinDelta = minDelta;
            return this;
        }
        
        /**
         * Enable model checkpointing.
         */
        public Builder withCheckpointing(String path) {
            this.enableCheckpointing = true;
            this.checkpointPath = path;
            return this;
        }
        
        /**
         * Enable model checkpointing with save strategy.
         */
        public Builder withCheckpointing(String path, boolean onlyBest) {
            this.enableCheckpointing = true;
            this.checkpointPath = path;
            this.checkpointOnlyBest = onlyBest;
            return this;
        }
        
        /**
         * Enable training visualization (plots).
         */
        public Builder withVisualization(String path) {
            this.enableVisualization = true;
            this.visualizationPath = path;
            return this;
        }
        
        /**
         * Set number of parallel batches to process concurrently.
         * 0 = auto-detect based on model complexity (default)
         * 1 = sequential processing
         * 2+ = specific number of concurrent batches
         */
        public Builder parallelBatches(int count) {
            batchConfigBuilder.parallelBatches(count);
            return this;
        }
        
        /**
         * Set learning rate schedule for training.
         * 
         * <p><b>Examples:</b>
         * <pre>{@code
         * // Cosine annealing with warmup
         * .withLearningRateSchedule(
         *     LearningRateSchedule.cosineAnnealing(0.001f, epochs, 5)
         * )
         * 
         * // Step decay
         * .withLearningRateSchedule(
         *     LearningRateSchedule.stepDecay(0.01f, 0.1f, 30)
         * )
         * }</pre>
         * 
         * @param schedule the learning rate schedule to use
         * @return this builder
         */
        public Builder withLearningRateSchedule(LearningRateSchedule schedule) {
            batchConfigBuilder.withLearningRateSchedule(schedule);
            return this;
        }
        
        /**
         * Build the configuration.
         */
        public SimpleNetTrainingConfig build() {
            return new SimpleNetTrainingConfig(this);
        }
    }
}