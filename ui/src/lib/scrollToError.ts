/** Bring the first invalid field into view and focus it. Call after a
 *  client-side validation failure so the error isn't off-screen. */
export function scrollToFirstError(root?: HTMLElement | null) {
  const scope = root ?? document
  const bad = scope.querySelector<HTMLElement>('[aria-invalid="true"], .hp-error')
  if (!bad) return
  bad.scrollIntoView({ behavior: 'smooth', block: 'center' })
  if (bad.matches('input,select,textarea')) bad.focus({ preventScroll: true })
  else bad.closest('.hp-field')?.querySelector<HTMLElement>('input,select,textarea')?.focus({ preventScroll: true })
}
