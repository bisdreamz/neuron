package dev.neuronic.net.training;

import dev.neuronic.net.NeuralNet;
import dev.neuronic.net.layers.Layer;
import dev.neuronic.net.training.TrainingMetrics.EpochMetrics;
import dev.neuronic.net.optimizers.Optimizer;

/**
 * Learning rate scheduler callback that adjusts learning rate during training.
 * 
 * @deprecated Use {@link LearningRateSchedule} with training configuration instead.
 * This callback modifies optimizer state as a side effect, which violates the
 * callback pattern. The new API provides cleaner separation of concerns:
 * <pre>{@code
 * // Instead of:
 * LearningRateSchedulerCallback scheduler = 
 *     LearningRateSchedulerCallback.cosineAnnealing(lr, epochs, warmup);
 * trainer.withCallback(scheduler);
 * 
 * // Use:
 * TrainingConfig config = TrainingConfig.builder()
 *     .withLearningRateSchedule(
 *         LearningRateSchedule.cosineAnnealing(lr, epochs, warmup)
 *     )
 *     .build();
 * }</pre>
 * 
 * Supports multiple scheduling strategies:
 * - Step decay: Reduce by factor every N epochs
 * - Exponential decay: Smooth exponential reduction
 * - Cosine annealing: Cosine-shaped reduction
 * - Reduce on plateau: Reduce when metric stops improving
 */
@Deprecated(since = "1.1", forRemoval = true)
public class LearningRateSchedulerCallback implements TrainingCallback {
    
    public enum ScheduleType {
        STEP_DECAY,
        EXPONENTIAL_DECAY,
        COSINE_ANNEALING,
        REDUCE_ON_PLATEAU
    }
    
    private final ScheduleType scheduleType;
    private final float initialLearningRate;
    private final float factor; // Reduction factor
    private final int patience; // For plateau detection
    private final int stepSize; // For step decay
    private final int warmupEpochs; // For cosine annealing warmup
    
    // State tracking
    private float currentLearningRate;
    private float bestValue;
    private int epochsWithoutImprovement;
    private boolean monitorIncreasing;
    private NeuralNet model; // Store model reference for optimizer updates
    
    /**
     * Creates a step decay scheduler.
     */
    public static LearningRateSchedulerCallback stepDecay(float initialLr, float factor, int stepSize) {
        return new LearningRateSchedulerCallback(ScheduleType.STEP_DECAY, initialLr, factor, 0, stepSize);
    }
    
    /**
     * Creates an exponential decay scheduler.
     */
    public static LearningRateSchedulerCallback exponentialDecay(float initialLr, float decayRate) {
        return new LearningRateSchedulerCallback(ScheduleType.EXPONENTIAL_DECAY, initialLr, decayRate, 0, 0);
    }
    
    /**
     * Creates a reduce-on-plateau scheduler.
     */
    public static LearningRateSchedulerCallback reduceOnPlateau(float initialLr, float factor, int patience) {
        return new LearningRateSchedulerCallback(ScheduleType.REDUCE_ON_PLATEAU, initialLr, factor, patience, 0);
    }
    
    /**
     * Creates a cosine annealing scheduler.
     */
    public static LearningRateSchedulerCallback cosineAnnealing(float initialLr, int totalEpochs) {
        return cosineAnnealing(initialLr, totalEpochs, 0);
    }
    
    /**
     * Creates a cosine annealing scheduler with warmup.
     * During warmup epochs, learning rate stays constant at initialLr.
     */
    public static LearningRateSchedulerCallback cosineAnnealing(float initialLr, int totalEpochs, int warmupEpochs) {
        return new LearningRateSchedulerCallback(ScheduleType.COSINE_ANNEALING, initialLr, 0, warmupEpochs, totalEpochs);
    }
    
    private LearningRateSchedulerCallback(ScheduleType type, float initialLr, 
                                         float factor, int patience, int stepSize) {
        this.scheduleType = type;
        this.initialLearningRate = initialLr;
        this.factor = factor;
        this.patience = patience;
        this.stepSize = stepSize;
        this.warmupEpochs = (type == ScheduleType.COSINE_ANNEALING) ? patience : 0; // Use patience as warmup for cosine
        this.currentLearningRate = initialLr;
    }
    
    @Override
    public void onTrainingStart(NeuralNet model, TrainingMetrics metrics) {
        this.model = model; // Store model reference
        currentLearningRate = initialLearningRate;
        epochsWithoutImprovement = 0;
        bestValue = Float.POSITIVE_INFINITY; // Monitor validation loss by default
        monitorIncreasing = false;
        
        System.out.printf("Learning rate scheduling: %s starting at lr=%.6f%n", 
                         scheduleType, initialLearningRate);
        
        // Apply initial learning rate
        updateModelLearningRate(model, currentLearningRate);
    }
    
    @Override
    public void onEpochEnd(int epoch, TrainingMetrics metrics) {
        EpochMetrics epochMetrics = metrics.getEpochMetrics(epoch);
        if (epochMetrics == null) return;
        
        float newLearningRate = currentLearningRate;
        
        switch (scheduleType) {
            case STEP_DECAY:
                if ((epoch + 1) % stepSize == 0) {
                    newLearningRate = currentLearningRate * factor;
                }
                break;
                
            case EXPONENTIAL_DECAY:
                newLearningRate = initialLearningRate * (float) Math.pow(factor, epoch);
                break;
                
            case COSINE_ANNEALING:
                // Check if we're still in warmup phase
                if (epoch < warmupEpochs) {
                    newLearningRate = initialLearningRate; // Keep constant during warmup
                } else {
                    // Cosine annealing after warmup
                    float adjustedEpoch = epoch - warmupEpochs;
                    float remainingEpochs = stepSize - warmupEpochs;
                    float progress = adjustedEpoch / remainingEpochs;
                    newLearningRate = initialLearningRate * 0.5f * 
                        (1 + (float) Math.cos(Math.PI * progress));
                    
                    // Ensure minimum learning rate to avoid "lr must be > 0" error
                    float minLearningRate = initialLearningRate * 0.001f; // 0.1% of initial
                    newLearningRate = Math.max(newLearningRate, minLearningRate);
                }
                break;
                
            case REDUCE_ON_PLATEAU:
                float currentValue = (float) epochMetrics.getValidationLoss();
                
                if (currentValue < bestValue - 1e-4) {
                    bestValue = currentValue;
                    epochsWithoutImprovement = 0;
                } else {
                    epochsWithoutImprovement++;
                    
                    if (epochsWithoutImprovement >= patience) {
                        newLearningRate = currentLearningRate * factor;
                        epochsWithoutImprovement = 0;
                        System.out.printf("Reducing learning rate to %.6f after %d epochs without improvement%n",
                                         newLearningRate, patience);
                    }
                }
                break;
        }
        
        // Update if changed
        if (Math.abs(newLearningRate - currentLearningRate) > 1e-8) {
            currentLearningRate = newLearningRate;
            
            // Update all optimizers with new learning rate
            if (model != null) {
                updateModelLearningRate(model, currentLearningRate);
            }
            
            if (scheduleType != ScheduleType.REDUCE_ON_PLATEAU) {
                System.out.printf("Epoch %d: learning rate adjusted to %.6f%n", 
                                 epoch + 1, currentLearningRate);
            }
        }
    }
    
    private void updateModelLearningRate(NeuralNet model, float learningRate) {
        // Update learning rate for all optimizers in the model
        for (Layer layer : model.getLayers()) {
            Optimizer optimizer = layer.getOptimizer();
            if (optimizer != null)
                optimizer.setLearningRate(learningRate);
        }
    }
    
    private void updateModelLearningRate(TrainingMetrics metrics, float learningRate) {
        // Can't access model from metrics, so this overload can't update optimizers
        // This is called from onEpochEnd where we only have metrics
        // TODO: Consider passing model to all callback methods
    }
}