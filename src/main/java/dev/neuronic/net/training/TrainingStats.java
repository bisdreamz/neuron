package dev.neuronic.net.training;

/**
 * Comprehensive training statistics with strongly-typed fields for easy discovery.
 * 
 * Provides all common metrics for both classification and language modeling tasks.
 * Users can type `stats.` and see all available metrics with proper documentation.
 */
public class TrainingStats {
    
    // Core metrics (always available)
    private final double trainLoss;
    private final double validationLoss;
    private final double trainAccuracy;
    private final double validationAccuracy;
    private final long trainingTimeMs;
    private final int epochsTrained;
    private final int totalSamples;
    private final int batchSize;
    
    // Language model specific metrics
    private final double trainPerplexity;
    private final double validationPerplexity;
    
    // Best epoch tracking
    private final int bestEpoch;
    private final double bestValidationLoss;
    private final double bestValidationAccuracy;
    private final double bestValidationPerplexity;
    
    // Vocabulary statistics (for models with embeddings)
    private final int vocabularySize;
    private final int uniqueTokens;
    private final double vocabularyCoverage;
    
    // Training configuration
    private final int parallelBatches;
    private final boolean earlyStopped;
    
    protected TrainingStats(Builder builder) {
        this.trainLoss = builder.trainLoss;
        this.validationLoss = builder.validationLoss;
        this.trainAccuracy = builder.trainAccuracy;
        this.validationAccuracy = builder.validationAccuracy;
        this.trainingTimeMs = builder.trainingTimeMs;
        this.epochsTrained = builder.epochsTrained;
        this.totalSamples = builder.totalSamples;
        this.batchSize = builder.batchSize;
        
        // Language model metrics
        this.trainPerplexity = builder.trainPerplexity;
        this.validationPerplexity = builder.validationPerplexity;
        
        // Best epoch
        this.bestEpoch = builder.bestEpoch;
        this.bestValidationLoss = builder.bestValidationLoss;
        this.bestValidationAccuracy = builder.bestValidationAccuracy;
        this.bestValidationPerplexity = builder.bestValidationPerplexity;
        
        // Vocabulary
        this.vocabularySize = builder.vocabularySize;
        this.uniqueTokens = builder.uniqueTokens;
        this.vocabularyCoverage = builder.vocabularyCoverage;
        
        // Configuration
        this.parallelBatches = builder.parallelBatches;
        this.earlyStopped = builder.earlyStopped;
    }
    
    // ===== Core Metrics =====
    
    /**
     * Final training loss.
     */
    public double getTrainLoss() {
        return trainLoss;
    }
    
    /**
     * Final validation loss.
     */
    public double getValidationLoss() {
        return validationLoss;
    }
    
    /**
     * Final training accuracy (0-1 for classification, may be low for language models).
     */
    public double getTrainAccuracy() {
        return trainAccuracy;
    }
    
    /**
     * Final validation accuracy (0-1 for classification, may be low for language models).
     */
    public double getValidationAccuracy() {
        return validationAccuracy;
    }
    
    /**
     * Total training time in milliseconds.
     */
    public long getTrainingTimeMs() {
        return trainingTimeMs;
    }
    
    /**
     * Total training time in seconds.
     */
    public double getTrainingTimeSeconds() {
        return trainingTimeMs / 1000.0;
    }
    
    /**
     * Number of epochs completed.
     */
    public int getEpochsTrained() {
        return epochsTrained;
    }
    
    /**
     * Total number of training samples.
     */
    public int getTotalSamples() {
        return totalSamples;
    }
    
    /**
     * Batch size used during training.
     */
    public int getBatchSize() {
        return batchSize;
    }
    
    // ===== Language Model Metrics =====
    
    /**
     * Training perplexity (exp(loss)). Lower is better.
     * For language models only - will be NaN for classification.
     */
    public double getTrainPerplexity() {
        return trainPerplexity;
    }
    
    /**
     * Validation perplexity (exp(loss)). Lower is better.
     * For language models only - will be NaN for classification.
     */
    public double getValidationPerplexity() {
        return validationPerplexity;
    }
    
    /**
     * Check if this is a language model based on perplexity availability.
     */
    public boolean isLanguageModel() {
        return !Double.isNaN(validationPerplexity);
    }
    
    // ===== Best Epoch Tracking =====
    
    /**
     * Epoch with best validation performance (0-indexed).
     */
    public int getBestEpoch() {
        return bestEpoch;
    }
    
    /**
     * Best validation loss achieved during training.
     */
    public double getBestValidationLoss() {
        return bestValidationLoss;
    }
    
    /**
     * Best validation accuracy achieved (for classification).
     */
    public double getBestValidationAccuracy() {
        return bestValidationAccuracy;
    }
    
    /**
     * Best validation perplexity achieved (for language models).
     */
    public double getBestValidationPerplexity() {
        return bestValidationPerplexity;
    }
    
    // ===== Vocabulary Statistics =====
    
    /**
     * Size of vocabulary (for models with embeddings).
     * Returns 0 if not applicable.
     */
    public int getVocabularySize() {
        return vocabularySize;
    }
    
    /**
     * Number of unique tokens in training data.
     * Returns 0 if not applicable.
     */
    public int getUniqueTokens() {
        return uniqueTokens;
    }
    
    /**
     * Percentage of tokens covered by vocabulary (0-100).
     * Returns 0 if not applicable.
     */
    public double getVocabularyCoverage() {
        return vocabularyCoverage;
    }
    
    // ===== Training Configuration =====
    
    /**
     * Number of parallel batches used during training.
     */
    public int getParallelBatches() {
        return parallelBatches;
    }
    
    
    /**
     * Whether training was stopped early.
     */
    public boolean wasEarlyStopped() {
        return earlyStopped;
    }
    
    // ===== Utility Methods =====
    
    /**
     * Get a human-readable summary of the training results.
     */
    public String getSummary() {
        StringBuilder sb = new StringBuilder();
        sb.append("=== Training Summary ===\n");
        sb.append(String.format("Training time: %.1fs\n", getTrainingTimeSeconds()));
        sb.append(String.format("Epochs trained: %d\n", epochsTrained));
        
        if (isLanguageModel()) {
            sb.append(String.format("Final perplexity: %.1f (train) / %.1f (val)\n", 
                                  trainPerplexity, validationPerplexity));
            sb.append(String.format("Best perplexity: %.1f (epoch %d)\n", 
                                  bestValidationPerplexity, bestEpoch + 1));
            
            // Interpretation
            if (validationPerplexity < 100) {
                sb.append("Model quality: Excellent - generates coherent text\n");
            } else if (validationPerplexity < 200) {
                sb.append("Model quality: Good - reasonable text generation\n");
            } else if (validationPerplexity < 500) {
                sb.append("Model quality: Fair - needs more training\n");
            } else {
                sb.append("Model quality: Poor - consider tuning hyperparameters\n");
            }
        } else {
            sb.append(String.format("Final accuracy: %.1f%% (train) / %.1f%% (val)\n", 
                                  trainAccuracy * 100, validationAccuracy * 100));
            sb.append(String.format("Best accuracy: %.1f%% (epoch %d)\n", 
                                  bestValidationAccuracy * 100, bestEpoch + 1));
        }
        
        if (vocabularySize > 0) {
            sb.append(String.format("Vocabulary: %,d words (%.1f%% coverage)\n", 
                                  vocabularySize, vocabularyCoverage));
        }
        
        if (earlyStopped) {
            sb.append("Training stopped early due to convergence\n");
        }
        
        return sb.toString();
    }
    
    /**
     * Print the summary to console.
     */
    public void printSummary() {
        System.out.println(getSummary());
    }
    
    // ===== Builder =====
    
    public static Builder builder() {
        return new Builder();
    }
    
    public static class Builder {
        protected double trainLoss;
        protected double validationLoss;
        protected double trainAccuracy;
        protected double validationAccuracy;
        protected long trainingTimeMs;
        protected int epochsTrained;
        protected int totalSamples;
        protected int batchSize;
        
        protected double trainPerplexity = Double.NaN;
        protected double validationPerplexity = Double.NaN;
        
        protected int bestEpoch;
        protected double bestValidationLoss;
        protected double bestValidationAccuracy;
        protected double bestValidationPerplexity = Double.NaN;
        
        protected int vocabularySize;
        protected int uniqueTokens;
        protected double vocabularyCoverage;
        
        protected int parallelBatches = 1;
        protected boolean earlyStopped;
        
        public Builder trainLoss(double trainLoss) {
            this.trainLoss = trainLoss;
            return this;
        }
        
        public Builder validationLoss(double validationLoss) {
            this.validationLoss = validationLoss;
            return this;
        }
        
        public Builder trainAccuracy(double trainAccuracy) {
            this.trainAccuracy = trainAccuracy;
            return this;
        }
        
        public Builder validationAccuracy(double validationAccuracy) {
            this.validationAccuracy = validationAccuracy;
            return this;
        }
        
        public Builder trainingTimeMs(long trainingTimeMs) {
            this.trainingTimeMs = trainingTimeMs;
            return this;
        }
        
        public Builder epochsTrained(int epochsTrained) {
            this.epochsTrained = epochsTrained;
            return this;
        }
        
        public Builder totalSamples(int totalSamples) {
            this.totalSamples = totalSamples;
            return this;
        }
        
        public Builder batchSize(int batchSize) {
            this.batchSize = batchSize;
            return this;
        }
        
        public Builder trainPerplexity(double trainPerplexity) {
            this.trainPerplexity = trainPerplexity;
            return this;
        }
        
        public Builder validationPerplexity(double validationPerplexity) {
            this.validationPerplexity = validationPerplexity;
            return this;
        }
        
        public Builder bestEpoch(int bestEpoch) {
            this.bestEpoch = bestEpoch;
            return this;
        }
        
        public Builder bestValidationLoss(double bestValidationLoss) {
            this.bestValidationLoss = bestValidationLoss;
            return this;
        }
        
        public Builder bestValidationAccuracy(double bestValidationAccuracy) {
            this.bestValidationAccuracy = bestValidationAccuracy;
            return this;
        }
        
        public Builder bestValidationPerplexity(double bestValidationPerplexity) {
            this.bestValidationPerplexity = bestValidationPerplexity;
            return this;
        }
        
        public Builder vocabularySize(int vocabularySize) {
            this.vocabularySize = vocabularySize;
            return this;
        }
        
        public Builder uniqueTokens(int uniqueTokens) {
            this.uniqueTokens = uniqueTokens;
            return this;
        }
        
        public Builder vocabularyCoverage(double vocabularyCoverage) {
            this.vocabularyCoverage = vocabularyCoverage;
            return this;
        }
        
        public Builder parallelBatches(int parallelBatches) {
            this.parallelBatches = parallelBatches;
            return this;
        }
        
        
        public Builder earlyStopped(boolean earlyStopped) {
            this.earlyStopped = earlyStopped;
            return this;
        }
        
        /**
         * Automatically calculate perplexity from loss if not set.
         */
        public Builder autoCalculatePerplexity() {
            if (Double.isNaN(trainPerplexity)) {
                trainPerplexity = Math.exp(trainLoss);
            }
            if (Double.isNaN(validationPerplexity)) {
                validationPerplexity = Math.exp(validationLoss);
            }
            if (Double.isNaN(bestValidationPerplexity)) {
                bestValidationPerplexity = Math.exp(bestValidationLoss);
            }
            return this;
        }
        
        public TrainingStats build() {
            return new TrainingStats(this);
        }
    }
}