let active: { owner: object; stop: () => void } | undefined;

export function claimSpeech(owner: object, stop: () => void) {
  const previous = active;
  active = undefined;
  previous?.stop();
  active = { owner, stop };
}

export function releaseSpeech(owner: object) {
  if (active?.owner === owner) active = undefined;
}
