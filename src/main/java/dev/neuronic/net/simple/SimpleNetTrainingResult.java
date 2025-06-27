package dev.neuronic.net.simple;

import dev.neuronic.net.training.TrainingMetrics;
import dev.neuronic.net.training.BatchTrainer;
import dev.neuronic.net.training.TrainingStats;
import dev.neuronic.net.training.ClassificationStats;
import dev.neuronic.net.training.RegressionStats;
import dev.neuronic.net.training.LanguageModelStats;
import java.io.IOException;
import java.util.List;

/**
 * Training result for SimpleNet advanced training.
 * Provides convenient access to training metrics and outcomes.
 */
public class SimpleNetTrainingResult {
    
    private final BatchTrainer.TrainingResult batchResult;
    private final long trainingTimeMs;
    private final int epochsTrained;
    
    public SimpleNetTrainingResult(BatchTrainer.TrainingResult batchResult, 
                                  long trainingTimeMs, int epochsTrained) {
        this.batchResult = batchResult;
        this.trainingTimeMs = trainingTimeMs;
        this.epochsTrained = epochsTrained;
    }
    
    /**
     * Get the underlying training metrics.
     */
    public TrainingMetrics getMetrics() {
        return batchResult.getMetrics();
    }
    
    /**
     * Get final training accuracy.
     */
    public float getFinalAccuracy() {
        return (float) batchResult.getMetrics().getFinalAccuracy();
    }
    
    /**
     * Get final training loss.
     */
    public float getFinalLoss() {
        double[] lossHistory = batchResult.getMetrics().getTrainingLossHistory();
        return lossHistory.length > 0 ? (float) lossHistory[lossHistory.length - 1] : 0.0f;
    }
    
    /**
     * Get final validation accuracy (if validation was used).
     */
    public float getFinalValidationAccuracy() {
        return (float) batchResult.getMetrics().getFinalValidationAccuracy();
    }
    
    /**
     * Get final validation loss (if validation was used).
     */
    public float getFinalValidationLoss() {
        double[] lossHistory = batchResult.getMetrics().getValidationLossHistory();
        return lossHistory.length > 0 ? (float) lossHistory[lossHistory.length - 1] : 0.0f;
    }
    
    /**
     * Get best validation accuracy achieved during training.
     */
    public float getBestValidationAccuracy() {
        return (float) batchResult.getMetrics().getBestValidationAccuracy();
    }
    
    /**
     * Get epoch number with best validation accuracy.
     */
    public int getBestEpoch() {
        return batchResult.getMetrics().getBestEpoch();
    }
    
    /**
     * Get total training time in milliseconds.
     */
    public long getTrainingTimeMs() {
        return trainingTimeMs;
    }
    
    /**
     * Get total training time as formatted string.
     */
    public String getTrainingTimeFormatted() {
        long seconds = trainingTimeMs / 1000;
        long minutes = seconds / 60;
        seconds = seconds % 60;
        
        if (minutes > 0) {
            return String.format("%dm %ds", minutes, seconds);
        } else {
            return String.format("%ds", seconds);
        }
    }
    
    /**
     * Get number of epochs actually trained (may be less than requested due to early stopping).
     */
    public int getEpochsTrained() {
        return epochsTrained;
    }
    
    /**
     * Check if training was stopped early.
     */
    public boolean wasStoppedEarly() {
        return epochsTrained < batchResult.getMetrics().getEpochCount();
    }
    
    /**
     * Export metrics to JSON file.
     */
    public void exportMetrics(String filename) throws IOException {
        batchResult.exportMetrics(filename);
    }
    
    /**
     * Print a summary of training results.
     */
    public void printSummary() {
        System.out.println("\n=== Training Summary ===");
        System.out.printf("Training time: %s%n", getTrainingTimeFormatted());
        System.out.printf("Epochs trained: %d%n", epochsTrained);
        
        if (wasStoppedEarly()) {
            System.out.println("Training stopped early");
        }
        
        System.out.printf("Final accuracy: %.2f%%%n", getFinalAccuracy() * 100);
        System.out.printf("Final loss: %.4f%n", getFinalLoss());
        
        if (getValidationLossHistory().length > 0) {
            System.out.printf("Final validation accuracy: %.2f%%%n", 
                            getFinalValidationAccuracy() * 100);
            System.out.printf("Final validation loss: %.4f%n", getFinalValidationLoss());
            System.out.printf("Best validation accuracy: %.2f%% (epoch %d)%n", 
                            getBestValidationAccuracy() * 100, getBestEpoch() + 1);
        }
        
        System.out.println("========================\n");
    }
    
    /**
     * Get training loss history.
     */
    public double[] getTrainingLossHistory() {
        return batchResult.getMetrics().getTrainingLossHistory();
    }
    
    /**
     * Get training accuracy history.
     */
    public double[] getTrainingAccuracyHistory() {
        return batchResult.getMetrics().getTrainingAccuracyHistory();
    }
    
    /**
     * Get validation loss history.
     */
    public double[] getValidationLossHistory() {
        return batchResult.getMetrics().getValidationLossHistory();
    }
    
    /**
     * Get validation accuracy history.
     */
    public double[] getValidationAccuracyHistory() {
        return batchResult.getMetrics().getValidationAccuracyHistory();
    }
    
    /**
     * Get comprehensive training statistics as a strongly-typed object.
     * This is the recommended way to access training results.
     * 
     * @return training statistics with all metrics
     */
    public TrainingStats getStats() {
        TrainingMetrics metrics = batchResult.getMetrics();
        
        // Detect if this is a language model
        boolean isLanguageModel = getFinalAccuracy() < 0.15 && getFinalValidationAccuracy() < 0.15;
        
        List<TrainingMetrics.EpochMetrics> allEpochs = metrics.getAllEpochMetrics();
        
        // Get best validation loss
        double bestValLoss = getFinalValidationLoss();
        for (TrainingMetrics.EpochMetrics epoch : allEpochs)
            if (epoch.getValidationLoss() < bestValLoss)
                bestValLoss = epoch.getValidationLoss();
        
        TrainingStats.Builder builder = TrainingStats.builder()
            .trainLoss(getFinalLoss())
            .validationLoss(getFinalValidationLoss())
            .trainAccuracy(getFinalAccuracy())
            .validationAccuracy(getFinalValidationAccuracy())
            .trainingTimeMs(trainingTimeMs)
            .epochsTrained(epochsTrained)
            .totalSamples(metrics.getTotalSamplesSeen())
            .batchSize(32)  // Default - could be passed from config
            .bestEpoch(getBestEpoch())
            .bestValidationLoss(bestValLoss)
            .bestValidationAccuracy(getBestValidationAccuracy())
            .earlyStopped(wasStoppedEarly());
        
        // Add perplexity for language models
        if (isLanguageModel) {
            builder.autoCalculatePerplexity();
        }
        
        return builder.build();
    }
    
    /**
     * Get type-specific training statistics.
     * Automatically detects the training type and returns the appropriate stats object.
     * 
     * @return ClassificationStats, RegressionStats, or LanguageModelStats
     */
    public TrainingStats getTypedStats() {
        TrainingMetrics metrics = batchResult.getMetrics();
        
        // Detect training type based on metrics
        boolean isLanguageModel = getFinalAccuracy() < 0.15 && getFinalValidationAccuracy() < 0.15;
        boolean isClassification = getFinalAccuracy() > 0.15; // Has meaningful accuracy
        
        List<TrainingMetrics.EpochMetrics> allEpochs = metrics.getAllEpochMetrics();
        
        // Get best validation loss
        double bestValLoss = getFinalValidationLoss();
        for (TrainingMetrics.EpochMetrics epoch : allEpochs)
            if (epoch.getValidationLoss() < bestValLoss)
                bestValLoss = epoch.getValidationLoss();
        
        if (isLanguageModel) {
            return LanguageModelStats.builder()
                .trainLoss(getFinalLoss())
                .validationLoss(getFinalValidationLoss())
                .trainAccuracy(getFinalAccuracy())
                .validationAccuracy(getFinalValidationAccuracy())
                .trainingTimeMs(trainingTimeMs)
                .epochsTrained(epochsTrained)
                .totalSamples(metrics.getTotalSamplesSeen())
                .batchSize(32)  // Default - could be passed from config
                .bestEpoch(getBestEpoch())
                .bestValidationLoss(bestValLoss)
                .bestValidationAccuracy(getBestValidationAccuracy())
                .bestValidationPerplexity(Math.exp(bestValLoss))
                    .earlyStopped(wasStoppedEarly())
                .autoCalculatePerplexity()
                .build();
        } else if (isClassification) {
            return ClassificationStats.builder()
                .trainLoss(getFinalLoss())
                .validationLoss(getFinalValidationLoss())
                .trainAccuracy(getFinalAccuracy())
                .validationAccuracy(getFinalValidationAccuracy())
                .trainingTimeMs(trainingTimeMs)
                .epochsTrained(epochsTrained)
                .totalSamples(metrics.getTotalSamplesSeen())
                .batchSize(32)  // Default - could be passed from config
                .bestEpoch(getBestEpoch())
                .bestValidationLoss(bestValLoss)
                .bestValidationAccuracy(getBestValidationAccuracy())
                    .earlyStopped(wasStoppedEarly())
                .build();
        } else {
            // Default to regression for continuous outputs
            RegressionStats.Builder regressionBuilder = (RegressionStats.Builder) RegressionStats.builder()
                .trainLoss(getFinalLoss())
                .validationLoss(getFinalValidationLoss())
                .trainAccuracy(getFinalAccuracy())
                .validationAccuracy(getFinalValidationAccuracy())
                .trainingTimeMs(trainingTimeMs)
                .epochsTrained(epochsTrained)
                .totalSamples(metrics.getTotalSamplesSeen())
                .batchSize(32)  // Default - could be passed from config
                .bestEpoch(getBestEpoch())
                .bestValidationLoss(bestValLoss)
                .bestValidationAccuracy(getBestValidationAccuracy())
                    .earlyStopped(wasStoppedEarly());
            
            return regressionBuilder.autoCalculateRMSE().build();
        }
    }
}