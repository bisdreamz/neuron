package dev.neuronic.net.training;

import dev.neuronic.net.NeuralNet;
import dev.neuronic.net.layers.Layer;
import dev.neuronic.net.optimizers.Optimizer;

/**
 * Learning rate monitoring callback that tracks and reports learning rate changes.
 * 
 * <p>This is a monitoring-only callback that observes learning rate changes
 * when using learning rate schedules. It does not modify the learning rate itself -
 * that's handled by the {@link LearningRateSchedule} configured in the training config.
 * 
 * <p><b>Usage:</b>
 * <pre>{@code
 * // Configure training with learning rate schedule
 * TrainingConfig config = TrainingConfig.builder()
 *     .withLearningRateSchedule(
 *         LearningRateSchedule.cosineAnnealing(0.001f, 100, 5)
 *     )
 *     .build();
 * 
 * // Add monitor to track learning rate changes
 * trainer.withCallback(new LearningRateMonitor());
 * }</pre>
 */
public class LearningRateMonitor implements TrainingCallback {
    
    private float currentLearningRate;
    private float previousLearningRate;
    
    @Override
    public void onTrainingStart(NeuralNet model, TrainingMetrics metrics) {
        // Get initial learning rate from the first layer's optimizer
        for (Layer layer : model.getLayers()) {
            Optimizer optimizer = layer.getOptimizer();
            if (optimizer != null) {
                // Note: We'd need a getLearningRate() method on Optimizer interface
                // For now, we'll just track changes based on what the schedule reports
                break;
            }
        }
        
        System.out.println("Learning rate monitoring enabled");
    }
    
    @Override
    public void onEpochEnd(int epoch, TrainingMetrics metrics) {
        // This callback is purely for monitoring - it doesn't change learning rates
        // The actual learning rate changes are handled by BatchTrainer when a
        // LearningRateSchedule is configured in the TrainingConfig
    }
    
    @Override
    public void onTrainingEnd(NeuralNet model, TrainingMetrics metrics) {
        System.out.println("Learning rate monitoring complete");
    }
    
    /**
     * Log learning rate change (called by BatchTrainer when schedule is applied).
     * 
     * @param epoch current epoch
     * @param newLearningRate new learning rate value
     */
    public void logLearningRateChange(int epoch, float newLearningRate) {
        if (Math.abs(newLearningRate - currentLearningRate) > 1e-8) {
            previousLearningRate = currentLearningRate;
            currentLearningRate = newLearningRate;
            
            System.out.printf("Epoch %d: learning rate changed from %.6f to %.6f%n", 
                             epoch + 1, previousLearningRate, currentLearningRate);
        }
    }
}