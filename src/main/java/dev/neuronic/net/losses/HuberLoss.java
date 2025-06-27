package dev.neuronic.net.losses;

import dev.neuronic.net.math.NetMath;

/**
 * Huber loss function - a robust alternative to MSE.
 * 
 * <p>The Huber loss combines the best properties of MSE and MAE:
 * <ul>
 *   <li>Quadratic for small errors (like MSE) - smooth gradients near zero
 *   <li>Linear for large errors (like MAE) - less sensitive to outliers
 * </ul>
 * 
 * <p>Loss definition:
 * <ul>
 *   <li>For |error| <= delta: loss = 0.5 * error^2
 *   <li>For |error| > delta: loss = delta * |error| - 0.5 * delta^2
 * </ul>
 * 
 * <p>Common delta values:
 * <ul>
 *   <li>1.0 - Default, good for most regression tasks
 *   <li>0.5 - More MSE-like behavior
 *   <li>2.0 - More MAE-like behavior
 * </ul>
 */
public final class HuberLoss implements Loss {
    
    private final float delta;
    
    /**
     * Create Huber loss with custom delta.
     * 
     * @param delta threshold parameter (must be positive)
     */
    public HuberLoss(float delta) {
        if (delta <= 0)
            throw new IllegalArgumentException("Delta must be positive, got: " + delta);
        
        this.delta = delta;
    }
    
    /**
     * Create Huber loss with default delta = 1.0
     */
    public static HuberLoss createDefault() {
        return new HuberLoss(1.0f);
    }
    
    /**
     * Create Huber loss with custom delta
     */
    public static HuberLoss create(float delta) {
        return new HuberLoss(delta);
    }
    
    @Override
    public float loss(float[] prediction, float[] labels) {
        return NetMath.lossComputeHuber(prediction, labels, delta);
    }
    
    @Override
    public float[] derivatives(float[] prediction, float[] labels) {
        float[] derivatives = new float[prediction.length];
        NetMath.lossDerivativesHuber(prediction, labels, delta, derivatives);
        return derivatives;
    }
    
    /**
     * Get the delta parameter
     */
    public float getDelta() {
        return delta;
    }
}