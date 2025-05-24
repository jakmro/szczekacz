import React, { useRef, useEffect, useState } from 'react';
import { AudioManager, AudioRecorder, AudioBuffer } from 'react-native-audio-api';
import { SafeAreaView, Text } from 'react-native';
import { useExecutorchModule } from 'react-native-executorch';
import Svg, { Rect, Text as SvgText } from 'react-native-svg';

// CONFIGURATION
const DOG_THRESHOLD = 0.5;
const CAT_THRESHOLD = 0.5;

// Uninstall the app and install the new version if you change the AUDIO_DURATION
const AUDIO_DURATION = 3;

// Softmax function to convert logits to probabilities
const softmax = (x: number[]) => {
  const max = Math.max(...x);
  const exps = x.map((xi) => Math.exp(xi - max));
  const sumExps = exps.reduce((a, b) => a + b, 0);
  return exps.map((xi) => xi / sumExps);
};

export default function App() {
  const model = useExecutorchModule({
    modelSource: require(`./assets/model_${AUDIO_DURATION}.pte`),
  });

  const recorderRef = useRef<AudioRecorder | null>(null);
  const audioBuffersRef = useRef<AudioBuffer[]>([]);

  const [probs, setProbs] = useState([0, 0, 0]);
  const [detectedClass, setDetectedClass] = useState<string>('None');

  useEffect(() => {
    AudioManager.setAudioSessionOptions({
      iosCategory: 'playAndRecord',
      iosMode: 'spokenAudio',
      iosOptions: ['allowBluetooth', 'defaultToSpeaker'],
    });

    recorderRef.current = new AudioRecorder({
      sampleRate: 16000,
      bufferLengthInSamples: 8000,
    });

    onRecord();
  }, [model.isReady]);

  const onRecord = () => {
    if (!recorderRef.current) {
      return;
    }
    console.log('Recording started');

    recorderRef.current.onAudioReady(async (buffer) => {
      audioBuffersRef.current = audioBuffersRef.current.slice(-5);
      audioBuffersRef.current.push(buffer);

      const waveform = Float32Array.of(
        ...audioBuffersRef.current[0].getChannelData(0),
        ...audioBuffersRef.current[1].getChannelData(0),
        ...audioBuffersRef.current[2].getChannelData(0),
        ...audioBuffersRef.current[3].getChannelData(0),
        ...audioBuffersRef.current[4].getChannelData(0),
        ...audioBuffersRef.current[5].getChannelData(0),
      );

      const output = await model.forward(waveform, [[1, 1, AUDIO_DURATION * 16000]]);
      const softmaxOutput = softmax(output[0]);

      setProbs(softmaxOutput);
      if (softmaxOutput[1] >= DOG_THRESHOLD) {
        setDetectedClass('Dog');
      } else if (softmaxOutput[0] >= CAT_THRESHOLD) {
        setDetectedClass('Cat');
      } else {
        setDetectedClass('None');
      }
    });

    recorderRef.current.onError((error) => {
      console.log('Audio recorder error:', error);
    });

    recorderRef.current.onStatusChange((status, previousStatus) => {
      console.log('Audio recorder status changed:', status, previousStatus);
    });

    recorderRef.current.start();
  };

  return (
    <SafeAreaView style={{ flex: 1, justifyContent: 'center', alignItems: 'center', backgroundColor: 'black' }}>
      <Svg height="200" width="300">
        {probs.map((prob, index) => (
          <React.Fragment key={index}>
            <Rect
              x={index * 100 + 20}
              y={200 - prob * 200}
              width={60}
              height={prob * 200}
              fill={['red', 'green', 'blue'][index]}
            />
            <SvgText
              x={index * 100 + 50}
              y={190}
              fontSize="16"
              fill="white"
              textAnchor="middle"
            >
              {['Cat', 'Dog', 'None'][index]}
            </SvgText>
          </React.Fragment>
        ))}
      </Svg>
      <Text style={{ fontSize: 24, color: 'white', marginTop: 20 }}>
        {`Detected: ${detectedClass}`}
      </Text>
    </SafeAreaView>
  );
}