"""Generate a Tableau workbook (.twb) with every worksheet pre-built.

Run it::

    uv run python tools/build_tableau_workbook.py

A .twb is XML, so the tedious part of dashboard building — wiring fields onto
shelves, creating calculated fields, choosing mark types — can be generated.
What is left for the GUI is the genuinely visual work: sorting, colour and
arranging sheets onto a dashboard, all of which are one or two clicks each.

Scope note: only elements verified to load are emitted. Tableau validates a
workbook against an internal schema on open and rejects the whole file for a
single unknown element, so speculative XML is expensive. An earlier attempt
included <simple-id>, which is not part of the 18.1 schema and blocked the
entire workbook.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CSV = ROOT / "data" / "exports" / "customers.csv"
OUT = ROOT / "tableau" / "churnguard.twb"

DS = "federated.churnguard"

# pandas dtype -> (tableau datatype, type, role)
TYPE_MAP = {
    "object": ("string", "nominal", "dimension"),
    "bool": ("boolean", "nominal", "dimension"),
    "int64": ("integer", "quantitative", "measure"),
    "float64": ("real", "quantitative", "measure"),
}

# Numeric columns that are really categories. Left as measures, Tableau would
# sum them, which is meaningless for a decile label or an add-on count.
FORCE_DIMENSION = {"risk_decile", "addon_service_count", "risk_rank"}

# Calculated fields. Rates are multiplied by 100 so the axis reads "42.7"
# without depending on a format string — Tableau's default-format='p1' rendered
# every axis tick as "1" when tried.
CALCULATIONS = {
    "Churn Rate %": 'AVG(IF [churned] = "Yes" THEN 100.0 ELSE 0.0 END)',
    "Revenue at Risk": "SUM([annual_revenue_lost])",
    "Customers": "COUNT([customer_id])",
    "Annual Revenue": "SUM([annual_revenue])",
    "Avg Monthly Charges": "AVG([monthly_charges])",
}


def escape(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")


def build_columns(frame: pd.DataFrame) -> tuple[str, str, str]:
    """Column declarations, metadata records and role overrides."""
    columns, metadata, overrides = [], [], []

    for i, name in enumerate(frame.columns):
        datatype, ttype, role = TYPE_MAP[str(frame[name].dtype)]
        columns.append(f"          <column datatype='{datatype}' name='{name}' ordinal='{i}' />")
        metadata.append(
            f"          <metadata-record class='column'>\n"
            f"            <remote-name>{name}</remote-name>\n"
            f"            <remote-type>{'130' if datatype == 'string' else '5'}</remote-type>\n"
            f"            <local-name>[{name}]</local-name>\n"
            f"            <parent-name>[customers.csv]</parent-name>\n"
            f"            <remote-alias>{name}</remote-alias>\n"
            f"            <ordinal>{i}</ordinal>\n"
            f"            <local-type>{datatype}</local-type>\n"
            f"            <aggregation>{'Count' if role == 'dimension' else 'Sum'}</aggregation>\n"
            f"            <contains-null>true</contains-null>\n"
            f"          </metadata-record>"
        )
        if name in FORCE_DIMENSION:
            overrides.append(
                f"      <column datatype='{datatype}' name='[{name}]' role='dimension' type='ordinal' />"
            )

    return "\n".join(columns), "\n".join(metadata), "\n".join(overrides)


def build_calculations() -> str:
    blocks = []
    for caption, formula in CALCULATIONS.items():
        blocks.append(
            f"      <column caption='{caption}' datatype='real' name='[{caption}]' "
            f"role='measure' type='quantitative'>\n"
            f"        <calculation class='tableau' formula='{escape(formula)}' />\n"
            f"      </column>"
        )
    return "\n".join(blocks)


def dependency(field: str, frame: pd.DataFrame) -> str:
    """One <column> entry inside a worksheet's datasource-dependencies."""
    if field in CALCULATIONS:
        return (
            f"            <column caption='{field}' datatype='real' name='[{field}]' "
            f"role='measure' type='quantitative'>\n"
            f"              <calculation class='tableau' formula='{escape(CALCULATIONS[field])}' />\n"
            f"            </column>"
        )
    datatype, ttype, role = TYPE_MAP[str(frame[field].dtype)]
    if field in FORCE_DIMENSION:
        role, ttype = "dimension", "ordinal"
    return f"            <column datatype='{datatype}' name='[{field}]' role='{role}' type='{ttype}' />"


def worksheet(
    name: str,
    rows: list[str],
    cols: list[str],
    frame: pd.DataFrame,
    mark: str = "Bar",
    color: str | None = None,
    label: str | None = None,
) -> str:
    """Emit one worksheet.

    Args:
        rows / cols: fields for the Rows and Columns shelves, in order.
        mark: Bar, Line, Text, Circle...
        color: field placed on the Color encoding.
        label: field placed on the Label encoding.
    """
    used = list(dict.fromkeys(rows + cols + ([color] if color else []) + ([label] if label else [])))
    deps = "\n".join(dependency(f, frame) for f in used)

    encodings = []
    if color:
        encodings.append(f"              <color column='[{DS}].[{color}]' />")
    if label:
        encodings.append(f"              <text column='[{DS}].[{label}]' />")
    encodings_xml = ("\n".join(encodings)) if encodings else ""
    encodings_block = f"            <encodings>\n{encodings_xml}\n            </encodings>\n" if encodings else ""

    rows_xml = " ".join(f"[{DS}].[{f}]" for f in rows)
    cols_xml = " ".join(f"[{DS}].[{f}]" for f in cols)

    return f"""    <worksheet name='{escape(name)}'>
      <table>
        <view>
          <datasources>
            <datasource caption='customers' name='{DS}' />
          </datasources>
          <datasource-dependencies datasource='{DS}'>
{deps}
          </datasource-dependencies>
          <aggregation value='true' />
        </view>
        <style />
        <panes>
          <pane selection-relaxation-option='selection-relaxation-allow'>
            <view>
              <breakdown value='auto' />
            </view>
            <mark class='{mark}' />
{encodings_block}          </pane>
        </panes>
        <rows>{rows_xml}</rows>
        <cols>{cols_xml}</cols>
      </table>
    </worksheet>"""


def window(name: str) -> str:
    return f"""    <window class='worksheet' name='{escape(name)}'>
      <cards>
        <edge name='left'>
          <strip size='160'>
            <card type='pages' />
            <card type='filters' />
            <card type='marks' />
          </strip>
        </edge>
        <edge name='top'>
          <strip size='2147483647'>
            <card type='columns' />
          </strip>
          <strip size='2147483647'>
            <card type='rows' />
          </strip>
        </edge>
      </cards>
    </window>"""


def main() -> None:
    frame = pd.read_csv(CSV, nrows=500)
    columns_xml, metadata_xml, overrides_xml = build_columns(frame)

    # Titles state the finding, not the field name — a chart title is the one
    # piece of text every reader looks at.
    sheets = [
        worksheet("1. Month-to-month churns 15x more than two-year",
                  rows=["contract_type"], cols=["Churn Rate %"], frame=frame,
                  label="Churn Rate %"),
        worksheet("2. Manual payment churns 3x more than automatic",
                  rows=["payment_method"], cols=["Churn Rate %"], frame=frame,
                  label="Churn Rate %"),
        worksheet("3. Over half of new customers leave within 6 months",
                  rows=["Churn Rate %"], cols=["tenure_band"], frame=frame,
                  mark="Line", label="Churn Rate %"),
        worksheet("4. Five segments carry most of the revenue lost",
                  rows=["contract_type", "payment_method"], cols=["Revenue at Risk"],
                  frame=frame, label="Revenue at Risk"),
        worksheet("5. Fiber optic churns twice as often as DSL",
                  rows=["internet_service"], cols=["Churn Rate %"], frame=frame,
                  label="Churn Rate %"),
        worksheet("6. Each add-on service makes customers stickier",
                  rows=["Churn Rate %"], cols=["addon_service_count"], frame=frame,
                  mark="Line", color="has_internet", label="Churn Rate %"),
        worksheet("7. Top risk decile churns at 70 percent, bottom at 1",
                  rows=["Churn Rate %"], cols=["risk_decile"], frame=frame,
                  label="Churn Rate %"),
        worksheet("8. Call list - who to contact this month",
                  rows=["risk_rank", "customer_id", "contract_type", "risk_tier"],
                  cols=["Avg Monthly Charges"], frame=frame, mark="Text",
                  label="Avg Monthly Charges"),
    ]
    names = [
        "1. Month-to-month churns 15x more than two-year",
        "2. Manual payment churns 3x more than automatic",
        "3. Over half of new customers leave within 6 months",
        "4. Five segments carry most of the revenue lost",
        "5. Fiber optic churns twice as often as DSL",
        "6. Each add-on service makes customers stickier",
        "7. Top risk decile churns at 70 percent, bottom at 1",
        "8. Call list - who to contact this month",
    ]

    twb = f"""<?xml version='1.0' encoding='utf-8' ?>
<workbook source-build='2023.3.0' source-platform='mac' version='18.1' xmlns:user='http://www.tableausoftware.com/xml/user'>
  <preferences>
    <preference name='ui.encoding.shelf.height' value='24' />
    <preference name='ui.shelf.height' value='26' />
  </preferences>
  <datasources>
    <datasource caption='customers' inline='true' name='{DS}' version='18.1'>
      <connection class='federated'>
        <named-connections>
          <named-connection caption='customers' name='textscan.churnguard'>
            <connection class='textscan' directory='{CSV.parent}' filename='{CSV.name}' password='' server='' />
          </named-connection>
        </named-connections>
        <relation connection='textscan.churnguard' name='customers.csv' table='[customers#csv]' type='table'>
          <columns character-set='UTF-8' header='yes' locale='en_US' separator=','>
{columns_xml}
          </columns>
        </relation>
        <metadata-records>
{metadata_xml}
        </metadata-records>
      </connection>
{overrides_xml}
{build_calculations()}
    </datasource>
  </datasources>
  <worksheets>
{chr(10).join(sheets)}
  </worksheets>
  <windows source-height='30'>
{chr(10).join(window(n) for n in names)}
  </windows>
</workbook>
"""

    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(twb, encoding="utf-8")

    import xml.etree.ElementTree as ET
    ET.parse(OUT)

    print(f"wrote {OUT.relative_to(ROOT)}  ({len(twb.splitlines())} lines)")
    print("XML well-formed ✓")
    print(f"worksheets: {len(sheets)}   calculated fields: {len(CALCULATIONS)}")


if __name__ == "__main__":
    main()
