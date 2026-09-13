# Optional local voice

Bellwether remains fully playable with text. Open **Voice options** and enable
voice for the current page. This preference resets on reload; nothing plays or
records automatically.

- **Read latest observation** reads only the latest observation already visible
  in your role's journal, using a browser voice reporting `localService=true`.
  Use **Stop audio** to interrupt. Private observations are never auto-read.
- **Dictate on this device** checks for an already installed language matching
  the browser's language. Only recognition supporting `processLocally=true`
  and `available({processLocally:true}) === 'available'` is started. It never
  falls back to remote recognition or automatically downloads a language pack.
- Finish or cancel recording, edit the separate transcript, then choose
  **Add to action draft**. This appends to any text you typed meanwhile.
  **Send action** is still a separate, manual step.
- Text and journal captions stay available when audio is unsupported or denied.
  Reconnection, role revocation, a changed turn, leaving the page or hiding the
  tab stops voice and discards unadded transcript. Your ordinary action draft
  retains its existing recovery behavior.

## Browser support and validation limits

On-device recognition is experimental and not a universal Web Speech feature.
MDN browser-compat-data inspected on 2026-09-09 lists `processLocally` for desktop
Chrome139+, but **not Chrome for Android**, Firefox or Safari. An Android browser
can expose ordinary speech recognition while lacking the on-device contract:
this example deliberately does not invoke that potentially remote service.
Typing therefore remains the Android dictation fallback until the browser has
the required local capability. Read-aloud is available only if that device
actually lists a local voice; a remote/default voice is never substituted.

References:
- [MDN on-device recognition](https://developer.mozilla.org/en-US/docs/Web/API/Web_Speech_API/Using_the_Web_Speech_API#on-device_speech_recognition)
- [processLocally](https://developer.mozilla.org/en-US/docs/Web/API/SpeechRecognition/processLocally)
- [localService](https://developer.mozilla.org/en-US/docs/Web/API/SpeechSynthesisVoice/localService)
- [Browser compatibility data](https://github.com/mdn/browser-compat-data/blob/main/api/SpeechRecognition.json)

The committed Chromium tests use explicitly mocked speech events to verify
opt-in, local-only selection, editable transcripts, permission/cancel errors and
revocation/late-callback isolation. A separate unmodified-browser probe records
API availability without starting a microphone or producing audio. Those are
software contract checks, **not** physical Pixel testing, audible-quality or
speech-recognition accuracy measurements.

## Data and scope

No microphone stream, audio file, or transcript is uploaded to an audio
provider by this example. Only the final manually submitted text goes through
the existing human-action API; the game may deliver that text to the intended
recipients and configured resident model as usual. Local speech depends on the
browser/OS honoring the Web Speech on-device contract. There is no voice cloning,
audio storage, speech identity, background listening or model audio understanding.

The coastal map is CSS/DOM presentation of the scenario's named locations and
the last resolved facility state. It is not satellite imagery or a model's
visual perception. Its button labels and journal supply the text equivalent.
This increment adds presentation/input accessibility, not audio/image reasoning
or evidence about real human behavior.
