interface Props {
  value: number;
  onChange: (value: number) => void;
  min?: number;
  max?: number;
  step?: number;
}

export default function NumberInput({
  value,
  onChange,
  min = 1,
  max = 9999,
  step = 1,
}: Props) {
  const increment = () => onChange(Math.min(value + step, max));
  const decrement = () => onChange(Math.max(value - step, min));

  return (
    <div className="number-input">
      <input
        type="number"
        value={value}
        min={min}
        max={max}
        onChange={(e) => {
          const v = parseInt(e.target.value);
          if (!isNaN(v)) onChange(Math.max(min, Math.min(max, v)));
        }}
      />
      <div className="number-input-arrows">
        <button type="button" onClick={increment} tabIndex={-1}>
          &#9650;
        </button>
        <button type="button" onClick={decrement} tabIndex={-1}>
          &#9660;
        </button>
      </div>
    </div>
  );
}
