package dev.neuronic.net.losses;

public interface Loss {

    public float loss(float[] prediction, float[] labels);

    public float[] derivatives(float[] prediction, float[] labels);

}
