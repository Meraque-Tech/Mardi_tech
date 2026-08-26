export const ACTIVE_POLL_MS = 2500;
export const IDLE_POLL_MS = 10000;
export const HIDDEN_POLL_MS = 30000;

export function nextStatusPollDelay({ hidden = false, busy = false } = {}) {
  if (hidden) {
    return HIDDEN_POLL_MS;
  }
  return busy ? ACTIVE_POLL_MS : IDLE_POLL_MS;
}
