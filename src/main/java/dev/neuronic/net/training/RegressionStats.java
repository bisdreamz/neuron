package dev.neuronic.net.training;

/**
 * Training statistics specific to regression tasks.
 * Provides regression-focused metrics with clear naming and documentation.
 */
public class RegressionStats extends TrainingStats {
    
    // Regression-specific metrics
    private final double trainMAE;  // Mean Absolute Error
    private final double validationMAE;
    private final double trainRMSE; // Root Mean Square Error
    private final double validationRMSE;
    private final double trainR2;   // R-squared (coefficient of determination)
    private final double validationR2;
    private final double trainMAPE; // Mean Absolute Percentage Error
    private final double validationMAPE;
    
    // Data characteristics
    private final double targetMean;
    private final double targetStdDev;
    private final double targetMin;
    private final double targetMax;
    
    private RegressionStats(Builder builder) {
        super(builder);
        this.trainMAE = builder.trainMAE;
        this.validationMAE = builder.validationMAE;
        this.trainRMSE = builder.trainRMSE;
        this.validationRMSE = builder.validationRMSE;
        this.trainR2 = builder.trainR2;
        this.validationR2 = builder.validationR2;
        this.trainMAPE = builder.trainMAPE;
        this.validationMAPE = builder.validationMAPE;
        this.targetMean = builder.targetMean;
        this.targetStdDev = builder.targetStdDev;
        this.targetMin = builder.targetMin;
        this.targetMax = builder.targetMax;
    }
    
    /**
     * Training Mean Absolute Error (MAE).
     * Average absolute difference between predicted and actual values.
     */
    public double getTrainMAE() {
        return trainMAE;
    }
    
    /**
     * Validation Mean Absolute Error (MAE).
     */
    public double getValidationMAE() {
        return validationMAE;
    }
    
    /**
     * Training Root Mean Square Error (RMSE).
     * Square root of average squared differences.
     */
    public double getTrainRMSE() {
        return trainRMSE;
    }
    
    /**
     * Validation Root Mean Square Error (RMSE).
     */
    public double getValidationRMSE() {
        return validationRMSE;
    }
    
    /**
     * Training R-squared (coefficient of determination).
     * Proportion of variance explained by the model (0-1, higher is better).
     */
    public double getTrainR2() {
        return trainR2;
    }
    
    /**
     * Validation R-squared.
     */
    public double getValidationR2() {
        return validationR2;
    }
    
    /**
     * Training Mean Absolute Percentage Error (MAPE).
     * Average percentage error (0-100%).
     */
    public double getTrainMAPE() {
        return trainMAPE;
    }
    
    /**
     * Validation Mean Absolute Percentage Error (MAPE).
     */
    public double getValidationMAPE() {
        return validationMAPE;
    }
    
    /**
     * Mean of target values in training data.
     */
    public double getTargetMean() {
        return targetMean;
    }
    
    /**
     * Standard deviation of target values.
     */
    public double getTargetStdDev() {
        return targetStdDev;
    }
    
    /**
     * Minimum target value.
     */
    public double getTargetMin() {
        return targetMin;
    }
    
    /**
     * Maximum target value.
     */
    public double getTargetMax() {
        return targetMax;
    }
    
    /**
     * Get normalized RMSE (RMSE / target range).
     * Useful for comparing models across different scales.
     */
    public double getNormalizedRMSE() {
        double range = targetMax - targetMin;
        return range > 0 ? validationRMSE / range : Double.NaN;
    }
    
    /**
     * Check if the model is better than a naive mean predictor.
     */
    public boolean isBetterThanBaseline() {
        return validationR2 > 0;
    }
    
    /**
     * Get the improvement over baseline (as percentage).
     */
    public double getImprovementOverBaseline() {
        return validationR2 * 100;
    }
    
    @Override
    public String getSummary() {
        StringBuilder sb = new StringBuilder();
        sb.append("=== Regression Training Summary ===\n");
        sb.append(String.format("Training time: %.1fs\n", getTrainingTimeSeconds()));
        sb.append(String.format("Epochs trained: %d\n", getEpochsTrained()));
        
        // Primary metrics
        sb.append(String.format("Final RMSE: %.4f (train) / %.4f (val)\n", 
                              trainRMSE, validationRMSE));
        sb.append(String.format("R² Score: %.3f (train) / %.3f (val)\n", 
                              trainR2, validationR2));
        
        // Additional metrics if available
        if (!Double.isNaN(validationMAE)) {
            sb.append(String.format("MAE: %.4f (train) / %.4f (val)\n",
                                  trainMAE, validationMAE));
        }
        
        if (!Double.isNaN(validationMAPE)) {
            sb.append(String.format("MAPE: %.1f%% (train) / %.1f%% (val)\n",
                                  trainMAPE, validationMAPE));
        }
        
        // Target statistics
        if (!Double.isNaN(targetMean)) {
            sb.append(String.format("Target range: [%.2f, %.2f], mean: %.2f\n",
                                  targetMin, targetMax, targetMean));
        }
        
        // Model quality assessment
        if (validationR2 >= 0.9) {
            sb.append("Model quality: Excellent fit\n");
        } else if (validationR2 >= 0.7) {
            sb.append("Model quality: Good fit\n");
        } else if (validationR2 >= 0.5) {
            sb.append("Model quality: Moderate fit\n");
        } else if (validationR2 > 0) {
            sb.append("Model quality: Poor fit - consider improving features/architecture\n");
        } else {
            sb.append("Model quality: Worse than baseline - review approach\n");
        }
        
        if (wasEarlyStopped())
            sb.append("Training stopped early due to convergence\n");
        
        return sb.toString();
    }
    
    public static Builder builder() {
        return new Builder();
    }
    
    public static class Builder extends TrainingStats.Builder {
        private double trainMAE = Double.NaN;
        private double validationMAE = Double.NaN;
        private double trainRMSE = Double.NaN;
        private double validationRMSE = Double.NaN;
        private double trainR2 = Double.NaN;
        private double validationR2 = Double.NaN;
        private double trainMAPE = Double.NaN;
        private double validationMAPE = Double.NaN;
        private double targetMean = Double.NaN;
        private double targetStdDev = Double.NaN;
        private double targetMin = Double.NaN;
        private double targetMax = Double.NaN;
        
        public Builder trainMAE(double trainMAE) {
            this.trainMAE = trainMAE;
            return this;
        }
        
        public Builder validationMAE(double validationMAE) {
            this.validationMAE = validationMAE;
            return this;
        }
        
        public Builder trainRMSE(double trainRMSE) {
            this.trainRMSE = trainRMSE;
            return this;
        }
        
        public Builder validationRMSE(double validationRMSE) {
            this.validationRMSE = validationRMSE;
            return this;
        }
        
        public Builder trainR2(double trainR2) {
            this.trainR2 = trainR2;
            return this;
        }
        
        public Builder validationR2(double validationR2) {
            this.validationR2 = validationR2;
            return this;
        }
        
        public Builder trainMAPE(double trainMAPE) {
            this.trainMAPE = trainMAPE;
            return this;
        }
        
        public Builder validationMAPE(double validationMAPE) {
            this.validationMAPE = validationMAPE;
            return this;
        }
        
        public Builder targetMean(double targetMean) {
            this.targetMean = targetMean;
            return this;
        }
        
        public Builder targetStdDev(double targetStdDev) {
            this.targetStdDev = targetStdDev;
            return this;
        }
        
        public Builder targetMin(double targetMin) {
            this.targetMin = targetMin;
            return this;
        }
        
        public Builder targetMax(double targetMax) {
            this.targetMax = targetMax;
            return this;
        }
        
        /**
         * Calculate RMSE from MSE (loss) if not set.
         */
        public Builder autoCalculateRMSE() {
            if (Double.isNaN(trainRMSE))
                trainRMSE = Math.sqrt(trainLoss);
            
            if (Double.isNaN(validationRMSE))
                validationRMSE = Math.sqrt(validationLoss);
            
            return this;
        }
        
        @Override
        public RegressionStats build() {
            return new RegressionStats(this);
        }
    }
}