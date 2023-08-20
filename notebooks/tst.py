from typing import cast, Any

import pandas as pd
from pandas import DataFrame, Series

from pasteur.transform import SeqTransformer, TransformerFactory, Transformer
from pasteur.module import ModuleFactory, get_module_dict, Module
from pasteur.attribute import Attribute, Attributes, SeqValue, get_dtype, SeqAttribute, GenAttribute
from pasteur.extras.transformers import DatetimeTransformer

from project.settings import PASTEUR_MODULES as modules


def _backref_cols(
    ids: pd.DataFrame, seq: pd.Series, data: pd.DataFrame | pd.Series, parent: str
):
    # Ref is calculated by mapping each id in data_df by merging its parent
    # key, sequence number to parent key, and the number - 1 and finding the
    # corresponding id for that row. Then, a join is performed.
    _IDX_NAME = "_id_lkjijk"
    _JOIN_NAME = "_id_zdjwk"
    ids_seq_prev = ids.join(seq + 1).reset_index(names=_JOIN_NAME)
    ids_seq = ids.join(seq, how="right").reset_index(names=_IDX_NAME)
    # FIXME: ids become float
    join_ids = ids_seq.merge(ids_seq_prev, on=[parent, seq.name], how='left').set_index(_IDX_NAME)[
        [_JOIN_NAME]
    ] # type: ignore
    ref_df = join_ids.join(data, on=_JOIN_NAME).drop(columns=_JOIN_NAME)
    ref_df.index.name = data.index.name
    if isinstance(data, pd.Series):
        return ref_df[data.name]
    return ref_df


def _calculate_seq(data: Series, parent: str, col_seq: str):
    _ID_SEQ = "_id_sdfasdf"
    seq = (
        cast(
            pd.Series,
            pd.concat({parent: ids[parent], _ID_SEQ: data}, axis=1)
            .groupby(parent)[_ID_SEQ]
            .rank("first"),
        )
        - 1
    )
    max_len = int(cast(float, seq.max())) + 1
    return seq.astype(get_dtype(max_len + 1)).rename(col_seq)


class SeqTransformerWrapper(SeqTransformer):
    name = "seqwrap"

    def __init__(
        self,
        modules: list[Module],
        ctx: dict[str, Any],
        seq: dict[str, Any],
        parent: str | None = None,
        seq_col: str | None = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.parent = parent
        self.seq_col_ref = seq_col

        # Load transformers
        assert ctx and seq
        ctx_kwargs = ctx.copy()
        ctx_type = ctx_kwargs.pop("type")
        self.ctx = get_module_dict(TransformerFactory, modules)[
            cast(str, ctx_type)
        ].build(**ctx_kwargs)
        assert isinstance(self.ctx, Transformer)

        seq_kwargs = seq.copy()
        seq_type = seq_kwargs.pop("type")
        self.seq = get_module_dict(TransformerFactory, modules)[
            cast(str, seq_type)
        ].build(**seq_kwargs)
        assert isinstance(self.seq, RefTransformer)

    def fit(
        self,
        table: str,
        data: Series | DataFrame,
        ref: dict[str, DataFrame],
        ids: DataFrame,
        seq_val: SeqValue | None = None,
        seq: Series | None = None,
    ) -> tuple[SeqValue, Series] | None:
        self.col = cast(str, data.name)
        self.table = table

        # Grab parent from seq_val if available
        if seq_val is not None:
            self.parent = seq_val.table
            self.col_seq = seq_val.name
        else:
            self.col_seq = f"{table}_seq"
        self.col_n = f'{table}_n'

        if not self.parent:
            # Infering parent through references
            self.parent = next(iter(ref))
        # Process references
        # if ref:
        #     self.ref_table = next(iter(ref))
        #     self.ref_col = cast(str, next(iter(ref[self.ref_table].keys())))

        assert (
            self.parent
        ), "Parent table not specified, use parameter 'parent' or a foreign reference."

        # If seq was not provided
        self.generate_seq = False
        if seq is None:
            self.generate_seq = True
            if isinstance(data, DataFrame):
                assert self.seq_col_ref is not None, f'Multiple columns are provided as input, specify which one is used sequence the table through parameter `seq_col`.'
                seq_col = data[self.seq_col_ref]
            else:
                seq_col = data
            seq = _calculate_seq(seq_col, self.parent, self.col_seq)
        self.max_len = cast(int, seq.max()) + 1

        ctx_data = (
            ids.join(data[seq == 0], how="right")
            .drop_duplicates(subset=[self.parent])
            .set_index(self.parent)[self.col]
        )
        if ref:
            ctx_ref = ids.drop_duplicates(subset=[self.parent])
            for name, ref_table in ref.items():
                ctx_ref = ctx_ref.join(ref_table, on=name, how="left")
            ctx_ref = ctx_ref.set_index(self.parent)

            assert isinstance(
                self.ctx, RefTransformer
            ), f"Reference found, initial transformer should be a reference transformer."
            self.ctx.fit(ctx_data, ctx_ref)
        else:
            self.ctx.fit(ctx_data)

        # Data series is all rows where seq > 0 (skip initial)
        ref_df = _backref_cols(ids, seq, data, self.parent)
        self.seq.fit(data, ref_df)

        # If a seq_val was not provided, assume seq was also none and
        # become the sequencer
        if seq_val is None:
            return SeqValue(self.col_seq, self.parent), cast(Series, seq)

    def reduce(self, other: "SeqTransformerWrapper"):
        self.ctx.reduce(other)
        self.seq.reduce(other)
        self.max_len = max(other.max_len, self.max_len)

    def transform(
        self,
        data: Series | DataFrame,
        ref: dict[str, DataFrame],
        ids: DataFrame,
        seq: Series | None = None,
    ) -> tuple[DataFrame, dict[str, DataFrame]] | tuple[
        DataFrame, dict[str, DataFrame], Series
    ]:
        parent = cast(str, self.parent)
        if self.generate_seq:
            if isinstance(data, DataFrame):
                assert self.seq_col_ref is not None, f'Multiple columns are provided as input, specify which one is used sequence the table through parameter `seq_col`.'
                seq_col = data[self.seq_col_ref]
            else:
                seq_col = data
            seq = _calculate_seq(seq_col, parent, self.col_seq)
        else:
            assert seq is not None

        ctx_data = (
            ids.join(data[seq == 0], how="right")
            .drop_duplicates(subset=[self.parent])
            .set_index(self.parent)[self.col]
        )
        if ref:
            ctx_ref = ids.drop_duplicates(subset=[self.parent])
            for name, ref_table in ref.items():
                ctx_ref = ctx_ref.join(ref_table, on=name, how="left")
            ctx_ref = ctx_ref.set_index(self.parent)

            if isinstance(ctx_ref, DataFrame) and ctx_ref.shape[1] == 1:
                ctx_ref = ctx_ref[next(iter(ctx_ref))]

            assert isinstance(
                self.ctx, RefTransformer
            ), f"Reference found, initial transformer should be a reference transformer."
            ctx = self.ctx.transform(ctx_data, ctx_ref)
        else:
            ctx = self.ctx.transform(ctx_data)

        # Data series is all rows where seq > 0 (skip initial)
        ref_df = _backref_cols(ids, seq, data, parent)
        enc = self.seq.transform(data, ref_df)

        if self.generate_seq:
            return enc, {parent: pd.concat([ctx, ids.join(seq).groupby(self.parent)[cast(str, seq.name)].max().rename(self.col_n) + 1], axis=1)}, seq
        return enc, {parent: ctx}


    def get_attributes(self) -> tuple[Attributes, dict[str, Attributes]]:
        return {
            self.col_seq: SeqAttribute(self.col_seq, cast(str, self.parent)),
            **self.seq.get_attributes(),
        }, {cast(str, self.parent): {**self.ctx.get_attributes(), self.col_n: GenAttribute(self.col_n, self.table, self.max_len)}}


s = SeqTransformerWrapper(modules, {"type": "datetime", "nullable": True}, {"type": "datetime", "nullable": True})
s.fit(
    "admissions", admissions["admittime"], {"patients": patients[["birth_year"]]}, ids
)
r = s.transform(admissions["admittime"], {"patients": patients[["birth_year"]]}, ids)
s.max_len
