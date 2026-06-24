"""Objective evaluation for optimization algorithms."""

from __future__ import annotations

from dataclasses import dataclass, field

from .models import DescriptorEmbeddingProvider, load_predictor, mic_to_efficacy
from .rri import calculate_rri


@dataclass
class SequenceObjectives:
    efficacy: float
    toxicity: float
    rri: float | None = None


@dataclass
class SequenceEvaluator:
    efficacy_predictor: object | None = None
    toxicity_predictor: object | None = None
    mic_predictor: object | None = None
    embedding_provider: object | None = None
    rri_max_mutants: int | None = None
    cache: dict[str, SequenceObjectives] = field(default_factory=dict)

    @classmethod
    def from_specs(
        cls,
        *,
        efficacy_model: str | None = None,
        toxicity_model: str | None = None,
        mic_predictor: str | None = None,
        embedding_model: str | None = None,
        rri_max_mutants: int | None = None,
    ) -> "SequenceEvaluator":
        return cls(
            efficacy_predictor=load_predictor(efficacy_model) if efficacy_model else None,
            toxicity_predictor=load_predictor(toxicity_model) if toxicity_model else None,
            mic_predictor=load_predictor(mic_predictor) if mic_predictor else None,
            embedding_provider=load_predictor(embedding_model) if embedding_model else None,
            rri_max_mutants=rri_max_mutants,
        )

    def objectives(self, sequence: str, *, include_rri: bool) -> SequenceObjectives:
        cache_key = f"{sequence}:{include_rri}"
        if cache_key in self.cache:
            return self.cache[cache_key]

        efficacy = self._efficacy(sequence)
        toxicity = self._toxicity(sequence)
        rri_score = None
        if include_rri:
            if self.mic_predictor is None:
                raise ValueError("RRI evaluation requires a MIC predictor")
            # Prefer learned embeddings when available, but keep the pipeline
            # usable with descriptor-space distances for lightweight runs.
            embedding = self.embedding_provider or self.efficacy_predictor or DescriptorEmbeddingProvider()
            rri_score = calculate_rri(
                sequence,
                self.mic_predictor,
                embedding,
                max_mutants=self.rri_max_mutants,
            ).rri
        result = SequenceObjectives(efficacy=efficacy, toxicity=toxicity, rri=rri_score)
        self.cache[cache_key] = result
        return result

    def fitness(self, sequence: str, *, include_rri: bool = False) -> float:
        objectives = self.objectives(sequence, include_rri=include_rri)
        if include_rri:
            assert objectives.rri is not None
            # The exponents mirror the paper's weighting while keeping every
            # objective on a multiplicative, non-negative scale.
            return (
                max(objectives.efficacy, 1e-12) ** 0.4
                * max(1.0 - objectives.toxicity, 1e-12) ** 0.3
                * max(objectives.rri, 1e-12) ** 0.3
            )
        return max(objectives.efficacy, 1e-12) ** 0.6 * max(1.0 - objectives.toxicity, 1e-12) ** 0.4

    def ga_baseline_fitness(self, sequence: str) -> float:
        objectives = self.objectives(sequence, include_rri=False)
        return objectives.efficacy - objectives.toxicity

    def _efficacy(self, sequence: str) -> float:
        if self.efficacy_predictor is not None:
            return float(self.efficacy_predictor.predict_proba([sequence])[0])
        if self.mic_predictor is None:
            raise ValueError("Efficacy evaluation requires an efficacy or MIC predictor")
        log10_mic = float(self.mic_predictor.predict([sequence])[0])
        return mic_to_efficacy(log10_mic)

    def _toxicity(self, sequence: str) -> float:
        if self.toxicity_predictor is None:
            raise ValueError("Toxicity evaluation requires a toxicity predictor")
        return float(self.toxicity_predictor.predict_proba([sequence])[0])
