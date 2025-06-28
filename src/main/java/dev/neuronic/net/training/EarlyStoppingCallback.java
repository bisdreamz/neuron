package dev.neuronic.net.training;

import dev.neuronic.net.NeuralNet;
import dev.neuronic.net.training.TrainingMetrics.EpochMetrics;
import java.util.concurrent.atomic.AtomicBoolean;

/**
 * Early stopping callback that monitors validation metrics and stops training
 * when the model stops improving.
 * 
 * Features:
 * - Monitors validation loss or accuracy
 * - Configurable patience (epochs without improvement)
 * - Minimum delta for considering improvement
 * - Can restore best weights (if checkpointing is enabled)
 */
public class EarlyStoppingCallback implements TrainingCallback {
    
    private final int patience;
    private final float minDelta;
    private final AtomicBoolean stopFlag;
    private final String monitor;
    private final boolean restoreBestWeights;
    
    // State tracking
    private float bestValue;
    private int epochsWithoutImprovement;
    private int bestEpoch;
    private boolean monitorIncreasing; // true if higher is better (e.g., accuracy)
    
    /**
     * Creates early stopping callback monitoring validation accuracy.
     */
    public EarlyStoppingCallback(int patience, float minDelta, AtomicBoolean stopFlag) {
        this(patience, minDelta, stopFlag, "val_accuracy", false);
    }
    
    /**
     * Creates early stopping callback for language models (monitors validation loss).
     */
    public static EarlyStoppingCallback forLanguageModel(int patience, float minDelta, AtomicBoolean stopFlag) {
        return new EarlyStoppingCallback(patience, minDelta, stopFlag, "val_loss", false);
    }
    
    /**
     * Full constructor with all options.
     * 
     * @param patience epochs to wait without improvement before stopping
     * @param minDelta minimum change to consider as improvement
     * @param stopFlag shared flag to signal training stop
     * @param monitor metric to monitor ("val_loss" or "val_accuracy")
     * @param restoreBestWeights whether to restore weights from best epoch
     */
    public EarlyStoppingCallback(int patience, float minDelta, AtomicBoolean stopFlag,
                                String monitor, boolean restoreBestWeights) {
        if (patience <= 0)
            throw new IllegalArgumentException("Patience must be positive");
        if (minDelta < 0)
            throw new IllegalArgumentException("Min delta must be non-negative");
        if (!monitor.equals("val_loss") && !monitor.equals("val_accuracy"))
            throw new IllegalArgumentException("Monitor must be 'val_loss' or 'val_accuracy'");
        
        this.patience = patience;
        this.minDelta = minDelta;
        this.stopFlag = stopFlag;
        this.monitor = monitor;
        this.restoreBestWeights = restoreBestWeights;
        
        // Determine if we want the metric to increase or decrease
        this.monitorIncreasing = monitor.equals("val_accuracy");
    }
    
    @Override
    public void onTrainingStart(NeuralNet model, TrainingMetrics metrics) {
        epochsWithoutImprovement = 0;
        bestEpoch = -1;
        
        // Initialize best value based on metric type
        bestValue = monitorIncreasing ? Float.NEGATIVE_INFINITY : Float.POSITIVE_INFINITY;
        
        System.out.printf("Early stopping: monitoring %s with patience=%d%n", 
                         monitor, patience);
    }
    
    @Override
    public void onEpochEnd(int epoch, TrainingMetrics metrics) {
        EpochMetrics epochMetrics = metrics.getEpochMetrics(epoch);
        if (epochMetrics == null) return;
        
        // Get monitored value
        float currentValue = monitor.equals("val_loss") 
            ? (float) epochMetrics.getValidationLoss() 
            : (float) epochMetrics.getValidationAccuracy();
        
        // Check for improvement
        boolean improved = false;
        if (monitorIncreasing) {
            // Higher is better (e.g., accuracy)
            improved = currentValue > bestValue + minDelta;
        } else {
            // Lower is better (e.g., loss)
            improved = currentValue < bestValue - minDelta;
        }
        
        if (improved) {
            bestValue = currentValue;
            bestEpoch = epoch;
            epochsWithoutImprovement = 0;
            
            if (restoreBestWeights) {
                // Save current weights (would need serialization support)
                // For now, just track the best epoch
            }
        } else {
            epochsWithoutImprovement++;
            
            if (epochsWithoutImprovement >= patience && !stopFlag.get()) {
                System.out.printf("%nEarly stopping triggered after %d epochs without improvement.%n", 
                                 patience);
                System.out.printf("Best %s: %.4f at epoch %d%n", 
                                 monitor, bestValue, bestEpoch + 1);
                
                stopFlag.set(true);
            }
        }
    }
    
    @Override
    public void onTrainingEnd(NeuralNet model, TrainingMetrics metrics) {
        if (bestEpoch >= 0) {
            System.out.printf("Training stopped. Best %s: %.4f at epoch %d%n", 
                             monitor, bestValue, bestEpoch + 1);
            
            if (restoreBestWeights && bestEpoch != metrics.getEpochCount() - 1) {
                System.out.println("Note: Model weights from best epoch would be restored if serialization was available");
            }
        }
    }
}