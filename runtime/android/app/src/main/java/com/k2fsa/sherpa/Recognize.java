package com.k2fsa.sherpa;

public class Recognize {
    static {
        System.loadLibrary("sherpa");
    }

    public static native void init(String modelDir);
    public static native void reset();
    public static native void acceptWaveform(float[] waveform);
    public static native void setInputFinished();
    public static native void startDecode();
    public static native String getResult();
}
