export function normalizeScore(entry) {
  const mateIn = entry?.mate_in ?? entry?.mateIn ?? null;
  const raw = typeof entry === "object" ? entry?.score : entry;
  let numeric = Number(raw);

  if (!Number.isFinite(numeric)) numeric = 0;

  if (mateIn !== null) {
    const sign = mateIn > 0 ? 1 : -1;
    const distance = Math.min(Math.abs(mateIn), 999);
    const sortScore = sign * (1000 - distance);
    return { numeric: sortScore, mateIn };
  }

  return { numeric, mateIn: null };
}

export function formatTopLineScore(entry) {
  const { numeric, mateIn } = normalizeScore(entry);

  if (mateIn !== null) {
    return {
      numeric,
      mateIn,
      label: `${mateIn > 0 ? "+" : "-"}M${Math.abs(mateIn)}`,
    };
  }

  return {
    numeric,
    mateIn: null,
    label: `${numeric >= 0 ? "+" : ""}${numeric.toFixed(2)}`,
  };
}

export function formatEvalBarScore(entry) {
  const { numeric, mateIn } = normalizeScore(entry);

  if (mateIn !== null) {
    return {
      numeric,
      mateIn,
      label: `M${Math.abs(mateIn)}`,
    };
  }

  return {
    numeric,
    mateIn: null,
    label: Math.abs(numeric).toFixed(1),
  };
}

export function formatLineWithMoveNumbers(lineSan, sideToMove, fullmoveNumber) {
  const tokens = (lineSan || "").trim().split(/\s+/).filter(Boolean);
  if (tokens.length === 0) return "";

  let moveNo = Number(fullmoveNumber) || 1;
  const parts = [];
  let idx = 0;

  if (sideToMove === "b") {
    if (tokens[idx]) {
      parts.push(`${moveNo}... ${tokens[idx]}`);
      idx += 1;
      moveNo += 1;
    }
  }

  while (idx < tokens.length) {
    const wMove = tokens[idx] || "";
    const bMove = tokens[idx + 1] || "";

    if (bMove) {
      parts.push(`${moveNo}. ${wMove} ${bMove}`);
      idx += 2;
    } else {
      parts.push(`${moveNo}. ${wMove}`);
      idx += 1;
    }

    moveNo += 1;
  }

  return parts.join(" ");
}