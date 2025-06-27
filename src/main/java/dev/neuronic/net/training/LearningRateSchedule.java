package dev.neuronic.net.training;

/**
 * Learning rate schedule interface for controlling learning rate during training.
 * 
 * <p>This interface provides a clean separation between training control (schedules)
 * and monitoring (callbacks). Learning rate schedules are explicitly configured
 * as part of the training configuration, making the behavior transparent and testable.
 * 
 * <p><b>Common Schedules:</b>
 * <ul>
 *   <li><b>Constant:</b> Fixed learning rate throughout training</li>
 *   <li><b>Step Decay:</b> Reduce by factor every N epochs</li>
 *   <li><b>Exponential Decay:</b> Smooth exponential reduction</li>
 *   <li><b>Cosine Annealing:</b> Cosine-shaped reduction with optional warmup</li>
 *   <li><b>Polynomial Decay:</b> Polynomial reduction to end learning rate</li>
 * </ul>
 * 
 * <p><b>Usage Example:</b>
 * <pre>{@code
 * // Cosine annealing with warmup
 * TrainingConfig config = TrainingConfig.builder()
 *     .epochs(100)
 *     .withLearningRateSchedule(
 *         LearningRateSchedule.cosineAnnealing(0.001f, 100, 5)
 *     )
 *     .build();
 * }</pre>
 */
public interface LearningRateSchedule {
    
    /**
     * Get the learning rate for a given epoch.
     * 
     * @param epoch current epoch (0-based)
     * @param totalEpochs total number of epochs in training
     * @return learning rate to use for this epoch
     */
    float getLearningRate(int epoch, int totalEpochs);
    
    // ===============================
    // FACTORY METHODS
    // ===============================
    
    /**
     * Constant learning rate throughout training.
     * 
     * @param learningRate the fixed learning rate
     * @return constant schedule
     */
    static LearningRateSchedule constant(float learningRate) {
        if (learningRate <= 0)
            throw new IllegalArgumentException("Learning rate must be positive");
            
        return (epoch, totalEpochs) -> learningRate;
    }
    
    /**
     * Step decay: reduce learning rate by factor every stepSize epochs.
     * 
     * <p><b>Example:</b> With initialLr=0.1, factor=0.1, stepSize=30:
     * <ul>
     *   <li>Epochs 0-29: lr = 0.1</li>
     *   <li>Epochs 30-59: lr = 0.01</li>
     *   <li>Epochs 60-89: lr = 0.001</li>
     * </ul>
     * 
     * @param initialLr initial learning rate
     * @param factor reduction factor (e.g., 0.1 for 10x reduction)
     * @param stepSize epochs between reductions
     * @return step decay schedule
     */
    static LearningRateSchedule stepDecay(float initialLr, float factor, int stepSize) {
        if (initialLr <= 0 || factor <= 0 || factor >= 1 || stepSize <= 0) {
            throw new IllegalArgumentException(
                "Invalid parameters: initialLr > 0, 0 < factor < 1, stepSize > 0");
        }
        
        return (epoch, totalEpochs) -> {
            int steps = epoch / stepSize;
            return initialLr * (float) Math.pow(factor, steps);
        };
    }
    
    /**
     * Exponential decay: smooth exponential reduction.
     * 
     * <p>Learning rate = initialLr * decayRate^epoch
     * 
     * @param initialLr initial learning rate
     * @param decayRate decay rate per epoch (e.g., 0.95)
     * @return exponential decay schedule
     */
    static LearningRateSchedule exponentialDecay(float initialLr, float decayRate) {
        if (initialLr <= 0 || decayRate <= 0 || decayRate >= 1) {
            throw new IllegalArgumentException(
                "Invalid parameters: initialLr > 0, 0 < decayRate < 1");
        }
        
        return (epoch, totalEpochs) -> 
            initialLr * (float) Math.pow(decayRate, epoch);
    }
    
    /**
     * Cosine annealing: cosine-shaped reduction to near zero.
     * 
     * <p>Gradually reduces learning rate following a cosine curve,
     * which provides smooth deceleration for better convergence.
     * 
     * @param initialLr initial learning rate
     * @param totalEpochs total epochs (needed for cosine calculation)
     * @return cosine annealing schedule
     */
    static LearningRateSchedule cosineAnnealing(float initialLr, int totalEpochs) {
        return cosineAnnealing(initialLr, totalEpochs, 0);
    }
    
    /**
     * Cosine annealing with warmup period.
     * 
     * <p>During warmup, learning rate stays constant at initialLr.
     * After warmup, follows cosine curve to near zero.
     * 
     * @param initialLr initial learning rate
     * @param totalEpochs total epochs for training
     * @param warmupEpochs epochs to maintain initial learning rate
     * @return cosine annealing schedule with warmup
     */
    static LearningRateSchedule cosineAnnealing(float initialLr, int totalEpochs, int warmupEpochs) {
        if (initialLr <= 0 || totalEpochs <= 0 || warmupEpochs < 0 || warmupEpochs >= totalEpochs) {
            throw new IllegalArgumentException(
                "Invalid parameters: initialLr > 0, totalEpochs > 0, 0 <= warmupEpochs < totalEpochs");
        }
        
        return (epoch, total) -> {
            if (epoch < warmupEpochs) {
                return initialLr;
            }
            
            float progress = (float)(epoch - warmupEpochs) / (totalEpochs - warmupEpochs);
            float cosineDecay = 0.5f * (1 + (float) Math.cos(Math.PI * progress));
            
            // Ensure minimum learning rate to avoid "lr must be > 0" errors
            float minLr = initialLr * 0.001f; // 0.1% of initial
            return Math.max(initialLr * cosineDecay, minLr);
        };
    }
    
    /**
     * Polynomial decay: polynomial reduction to end learning rate.
     * 
     * <p>lr = (initialLr - endLr) * (1 - epoch/totalEpochs)^power + endLr
     * 
     * @param initialLr initial learning rate
     * @param endLr final learning rate
     * @param totalEpochs total epochs
     * @param power polynomial power (1.0 = linear, 2.0 = quadratic)
     * @return polynomial decay schedule
     */
    static LearningRateSchedule polynomialDecay(float initialLr, float endLr, int totalEpochs, float power) {
        if (initialLr <= 0 || endLr < 0 || endLr >= initialLr || totalEpochs <= 0 || power <= 0) {
            throw new IllegalArgumentException(
                "Invalid parameters: initialLr > 0, 0 <= endLr < initialLr, totalEpochs > 0, power > 0");
        }
        
        return (epoch, total) -> {
            if (epoch >= totalEpochs) {
                return endLr;
            }
            
            float progress = (float) epoch / totalEpochs;
            float decay = (float) Math.pow(1 - progress, power);
            return (initialLr - endLr) * decay + endLr;
        };
    }
    
    /**
     * Linear warmup: gradually increase from 0 to target learning rate.
     * 
     * <p>Useful for stable training start, especially with large learning rates.
     * 
     * @param targetLr target learning rate after warmup
     * @param warmupEpochs epochs to reach target
     * @return linear warmup schedule
     */
    static LearningRateSchedule linearWarmup(float targetLr, int warmupEpochs) {
        if (targetLr <= 0 || warmupEpochs <= 0) {
            throw new IllegalArgumentException(
                "Invalid parameters: targetLr > 0, warmupEpochs > 0");
        }
        
        return (epoch, totalEpochs) -> {
            if (epoch >= warmupEpochs) {
                return targetLr;
            }
            return targetLr * ((float)(epoch + 1) / warmupEpochs);
        };
    }
    
    /**
     * Combine two schedules: use first schedule until switchEpoch, then second.
     * 
     * <p>Useful for warmup followed by decay, or other custom combinations.
     * 
     * @param first first schedule to use
     * @param second second schedule to use
     * @param switchEpoch epoch to switch schedules
     * @return combined schedule
     */
    static LearningRateSchedule combine(LearningRateSchedule first, LearningRateSchedule second, int switchEpoch) {
        if (first == null || second == null || switchEpoch < 0) {
            throw new IllegalArgumentException(
                "Invalid parameters: schedules must be non-null, switchEpoch >= 0");
        }
        
        return (epoch, totalEpochs) -> {
            if (epoch < switchEpoch) {
                return first.getLearningRate(epoch, totalEpochs);
            } else {
                return second.getLearningRate(epoch, totalEpochs);
            }
        };
    }
}