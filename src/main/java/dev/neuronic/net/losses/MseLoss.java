package dev.neuronic.net.losses;

import dev.neuronic.net.math.NetMath;

/**
 * Mean Squared Error (MSE) loss function.
 * 
 * <p>Loss: MSE = (1/n) * sum((prediction[i] - target[i])^2)
 * <p>Derivative: dMSE/dprediction[i] = (2/n) * (prediction[i] - target[i])
 * 
 * <p>Commonly used for regression tasks. Penalizes larger errors more heavily
 * than smaller ones due to the squared term.
 */
public final class MseLoss implements Loss {
    
    public static final MseLoss INSTANCE = new MseLoss();
    
    private MseLoss() {} // Private constructor for singleton
    
    @Override
    public float loss(float[] prediction, float[] labels) {
        return NetMath.lossComputeMSE(prediction, labels);
    }
    
    @Override
    public float[] derivatives(float[] prediction, float[] labels) {
        float[] derivatives = new float[prediction.length];

        NetMath.lossDerivativesMSE(prediction, labels, derivatives);

        return derivatives;
    }
}