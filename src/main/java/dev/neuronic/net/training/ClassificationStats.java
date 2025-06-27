package dev.neuronic.net.training;

/**
 * Training statistics specific to classification tasks.
 * Provides classification-focused metrics with clear naming and documentation.
 */
public class ClassificationStats extends TrainingStats {
    
    // Classification-specific metrics
    private final double trainPrecision;
    private final double validationPrecision;
    private final double trainRecall;
    private final double validationRecall;
    private final double trainF1Score;
    private final double validationF1Score;
    private final int numClasses;
    private final double[] classDistribution;
    
    private ClassificationStats(Builder builder) {
        super(builder);
        this.trainPrecision = builder.trainPrecision;
        this.validationPrecision = builder.validationPrecision;
        this.trainRecall = builder.trainRecall;
        this.validationRecall = builder.validationRecall;
        this.trainF1Score = builder.trainF1Score;
        this.validationF1Score = builder.validationF1Score;
        this.numClasses = builder.numClasses;
        this.classDistribution = builder.classDistribution;
    }
    
    /**
     * Training precision (true positives / (true positives + false positives)).
     */
    public double getTrainPrecision() {
        return trainPrecision;
    }
    
    /**
     * Validation precision.
     */
    public double getValidationPrecision() {
        return validationPrecision;
    }
    
    /**
     * Training recall (true positives / (true positives + false negatives)).
     */
    public double getTrainRecall() {
        return trainRecall;
    }
    
    /**
     * Validation recall.
     */
    public double getValidationRecall() {
        return validationRecall;
    }
    
    /**
     * Training F1 score (harmonic mean of precision and recall).
     */
    public double getTrainF1Score() {
        return trainF1Score;
    }
    
    /**
     * Validation F1 score.
     */
    public double getValidationF1Score() {
        return validationF1Score;
    }
    
    /**
     * Number of classes in the classification problem.
     */
    public int getNumClasses() {
        return numClasses;
    }
    
    /**
     * Distribution of classes in training data (percentage for each class).
     */
    public double[] getClassDistribution() {
        return classDistribution;
    }
    
    /**
     * Check if the classes are balanced (no class has less than 10% of average).
     */
    public boolean isBalanced() {
        if (classDistribution == null || classDistribution.length == 0)
            return true;
        
        double avgPerClass = 100.0 / numClasses;
        double threshold = avgPerClass * 0.1; // 10% of average
        
        for (double classPercent : classDistribution)
            if (classPercent < threshold)
                return false;
        
        return true;
    }
    
    /**
     * Get training accuracy as percentage (0-100).
     */
    public double getTrainAccuracyPercent() {
        return getTrainAccuracy() * 100;
    }
    
    /**
     * Get validation accuracy as percentage (0-100).
     */
    public double getValidationAccuracyPercent() {
        return getValidationAccuracy() * 100;
    }
    
    /**
     * Get best validation accuracy as percentage (0-100).
     */
    public double getBestValidationAccuracyPercent() {
        return getBestValidationAccuracy() * 100;
    }
    
    @Override
    public String getSummary() {
        StringBuilder sb = new StringBuilder();
        sb.append("=== Classification Training Summary ===\n");
        sb.append(String.format("Training time: %.1fs\n", getTrainingTimeSeconds()));
        sb.append(String.format("Epochs trained: %d\n", getEpochsTrained()));
        
        // Accuracy
        sb.append(String.format("Final accuracy: %.1f%% (train) / %.1f%% (val)\n", 
                              getTrainAccuracyPercent(), 
                              getValidationAccuracyPercent()));
        sb.append(String.format("Best accuracy: %.1f%% (epoch %d)\n", 
                              getBestValidationAccuracyPercent(), 
                              getBestEpoch() + 1));
        
        // Additional metrics if available
        if (!Double.isNaN(validationF1Score)) {
            sb.append(String.format("F1 Score: %.3f (train) / %.3f (val)\n",
                                  trainF1Score, validationF1Score));
        }
        
        // Class information
        if (numClasses > 0) {
            sb.append(String.format("Classes: %d", numClasses));
            if (!isBalanced())
                sb.append(" (imbalanced dataset)");
            sb.append("\n");
        }
        
        // Model quality assessment
        if (getValidationAccuracyPercent() >= 95) {
            sb.append("Model quality: Excellent\n");
        } else if (getValidationAccuracyPercent() >= 85) {
            sb.append("Model quality: Good\n");
        } else if (getValidationAccuracyPercent() >= 70) {
            sb.append("Model quality: Fair - consider more training\n");
        } else {
            sb.append("Model quality: Poor - review model architecture\n");
        }
        
        if (wasEarlyStopped())
            sb.append("Training stopped early due to convergence\n");
        
        return sb.toString();
    }
    
    public static Builder builder() {
        return new Builder();
    }
    
    public static class Builder extends TrainingStats.Builder {
        private double trainPrecision = Double.NaN;
        private double validationPrecision = Double.NaN;
        private double trainRecall = Double.NaN;
        private double validationRecall = Double.NaN;
        private double trainF1Score = Double.NaN;
        private double validationF1Score = Double.NaN;
        private int numClasses;
        private double[] classDistribution;
        
        public Builder trainPrecision(double trainPrecision) {
            this.trainPrecision = trainPrecision;
            return this;
        }
        
        public Builder validationPrecision(double validationPrecision) {
            this.validationPrecision = validationPrecision;
            return this;
        }
        
        public Builder trainRecall(double trainRecall) {
            this.trainRecall = trainRecall;
            return this;
        }
        
        public Builder validationRecall(double validationRecall) {
            this.validationRecall = validationRecall;
            return this;
        }
        
        public Builder trainF1Score(double trainF1Score) {
            this.trainF1Score = trainF1Score;
            return this;
        }
        
        public Builder validationF1Score(double validationF1Score) {
            this.validationF1Score = validationF1Score;
            return this;
        }
        
        public Builder numClasses(int numClasses) {
            this.numClasses = numClasses;
            return this;
        }
        
        public Builder classDistribution(double[] classDistribution) {
            this.classDistribution = classDistribution;
            return this;
        }
        
        /**
         * Calculate F1 score from precision and recall if not set.
         */
        public Builder autoCalculateF1() {
            if (Double.isNaN(trainF1Score) && !Double.isNaN(trainPrecision) && !Double.isNaN(trainRecall)) {
                trainF1Score = 2 * (trainPrecision * trainRecall) / (trainPrecision + trainRecall);
            }
            if (Double.isNaN(validationF1Score) && !Double.isNaN(validationPrecision) && !Double.isNaN(validationRecall)) {
                validationF1Score = 2 * (validationPrecision * validationRecall) / (validationPrecision + validationRecall);
            }
            return this;
        }
        
        @Override
        public ClassificationStats build() {
            return new ClassificationStats(this);
        }
    }
}