package dev.neuronic.net.layers;

import java.util.List;

/**
 * An interface for layers that manage their own gradients (like embedding layers)
 * and need to participate in global gradient clipping.
 *
 * This allows the NeuralNet to retrieve all gradients for norm calculation,
 * apply a scaling factor if clipping is needed, and then let the layer
 * apply its own scaled gradients.
 */
public interface GradientProvider {

    /**
     * Gets the gradients that were updated in the last backward pass.
     *
     * For sparse layers like embeddings, this should only return the gradients
     * for the weights that were actually touched, not the entire gradient table.
     *
     * @return A list of 2D float arrays representing the gradients.
     */
    List<float[][]> getGradients();

    /**
     * Applies a scaling factor to the layer's internal gradients.
     * This is called by the training loop when global gradient clipping is applied.
     *
     * @param scale The scaling factor to apply (e.g., maxNorm / actualNorm).
     */
    void applyClippingScale(float scale);
}
