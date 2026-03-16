'''import os
import pandas as pd
from dotenv import load_dotenv, find_dotenv
from neo4j import GraphDatabase
from sentence_transformers import SentenceTransformer

# Load .env
load_dotenv(find_dotenv())

# Neo4j connection
NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))

# Paths: update to the real location of the CSVs
base_path = "D:/be proj(sinchan)"  # or an absolute path

patients_df = pd.read_csv(f"{base_path}/patients.csv")
doctors_df = pd.read_csv(f"{base_path}/doctors.csv")
appointments_df = pd.read_csv(f"{base_path}/appointments.csv")
treatments_df = pd.read_csv(f"{base_path}/treatments.csv")
billing_df = pd.read_csv(f"{base_path}/billing.csv")

# Normalize column names
dfs = [patients_df, doctors_df, appointments_df, treatments_df, billing_df]
for df in dfs:
    df.columns = (
        df.columns.astype(str)
        .str.strip()
        .str.lower()
        .str.replace(" ", "_")
    )

# Coerce and clean key date/time fields
# Patients: registration_date, date_of_birth
if "registrationdate" in patients_df.columns:
    patients_df["registrationdate"] = pd.to_datetime(
        patients_df["registrationdate"], errors="coerce"
    ).dt.date
if "dateofbirth" in patients_df.columns:
    patients_df["dateofbirth"] = pd.to_datetime(
        patients_df["dateofbirth"], errors="coerce"
    ).dt.date

# Appointments: appointmentdate, appointmenttime (normalize to ISO if present)
if "appointmentdate" in appointments_df.columns:
    appointments_df["appointmentdate"] = pd.to_datetime(
        appointments_df["appointmentdate"], errors="coerce"
    ).dt.date
if "appointmenttime" in appointments_df.columns:
    # appointmenttime appears as HHMMSS or similar strings;
    # standardize to HH:MM:SS
    def fix_time(val):
        s = str(val).zfill(6)
        return (
            f"{s[0:2]}:{s[2:4]}:{s[4:6]}"
            if s.isdigit() and len(s) == 6
            else None
        )
    appointments_df["appointmenttime"] = appointments_df[
        "appointmenttime"
    ].apply(fix_time)

# Treatments: treatmentdate, cost numeric
if "treatmentdate" in treatments_df.columns:
    treatments_df["treatmentdate"] = pd.to_datetime(
        treatments_df["treatmentdate"], errors="coerce"
    ).dt.date
if "cost" in treatments_df.columns:
    treatments_df["cost"] = pd.to_numeric(
        treatments_df["cost"], errors="coerce"
    )

# Billing: billdate, amount numeric
if "billdate" in billing_df.columns:
    billing_df["billdate"] = pd.to_datetime(
        billing_df["billdate"], errors="coerce"
    ).dt.date
if "amount" in billing_df.columns:
    billing_df["amount"] = pd.to_numeric(
        billing_df["amount"], errors="coerce"
    )

# Cypher helpers
def create_node(tx, label, id_key, row):
    # Convert numpy types to native; also drop NaNs to avoid setting nulls
    props = {
        k: (
            None
            if pd.isna(v)
            else (float(v) if isinstance(v, (pd.Float64Dtype, float)) else v)
        )
        for k, v in dict(row).items()
    }
    # Filter None values so SET n += doesn't store them
    props = {k: v for k, v in props.items() if v is not None}
    tx.run(
        f"MERGE (n:{label} {{{id_key}: $id}}) SET n += $props",
        id=row[id_key],
        props=props,
    )


def create_relationship(tx, query, params):
    tx.run(query, **params)


# Build graph
  
with driver.session() as session:
    # Patients
    for _, row in patients_df.iterrows():
        session.execute_write(create_node, "Patient", "patientid", row)

    # Doctors
    for _, row in doctors_df.iterrows():
        session.execute_write(create_node, "Doctor", "doctorid", row)

    # Appointments and relationships:
    # (Patient)-[HAS_APPOINTMENT]->(Appointment)<-[PERFORMED_BY]-(Doctor)
    for _, row in appointments_df.iterrows():
        session.execute_write(create_node, "Appointment", "appointmentid", row)
        # Link to Patient
        if pd.notna(row.get("patientid")):
            session.execute_write(
                create_relationship,
                (
                    """
                    MATCH (p:Patient {patientid: $patientid})
                    MATCH (a:Appointment {appointmentid: $appointmentid})
                    MERGE (p)-[:HAS_APPOINTMENT]->(a)
                    """
                ),
                {
                    "patientid": row["patientid"],
                    "appointmentid": row["appointmentid"],
                },
            )
        # Link to Doctor
        if pd.notna(row.get("doctorid")):
            session.execute_write(
                create_relationship,
                (
                    """
                    MATCH (d:Doctor {doctorid: $doctorid})
                    MATCH (a:Appointment {appointmentid: $appointmentid})
                    MERGE (d)-[:PERFORMED_BY]->(a)
                    """
                ),
                {
                    "doctorid": row["doctorid"],
                    "appointmentid": row["appointmentid"],
                },
            )

    # Treatments and relationships: (Appointment)-[HAS_TREATMENT]->(Treatment)
    for _, row in treatments_df.iterrows():
        session.execute_write(create_node, "Treatment", "treatmentid", row)
        if pd.notna(row.get("appointmentid")):
            session.execute_write(
                create_relationship,
                (
                    """
                    MATCH (ap:Appointment {appointmentid: $appointmentid})
                    MATCH (t:Treatment {treatmentid: $treatmentid})
                    MERGE (ap)-[:HAS_TREATMENT]->(t)
                    """
                ),
                {
                    "appointmentid": row["appointmentid"],
                    "treatmentid": row["treatmentid"],
                },
            )

    # Billing and relationships:
    # (Patient)-[HAS_BILL]->(Bill) and (Treatment)-[BILLED_AS]->(Bill)
    for _, row in billing_df.iterrows():
        session.execute_write(create_node, "Bill", "billid", row)
        if pd.notna(row.get("patientid")):
            session.execute_write(
                create_relationship,
                (
                    """
                    MATCH (p:Patient {patientid: $patientid})
                    MATCH (b:Bill {billid: $billid})
                    MERGE (p)-[:HAS_BILL]->(b)
                    """
                ),
                {
                    "patientid": row["patientid"],
                    "billid": row["billid"],
                },
            )
        if pd.notna(row.get("treatmentid")):
            session.execute_write(
                create_relationship,
                (
                    """
                    MATCH (t:Treatment {treatmentid: $treatmentid})
                    MATCH (b:Bill {billid: $billid})
                    MERGE (t)-[:BILLED_AS]->(b)
                    """
                ),
                {
                    "treatmentid": row["treatmentid"],
                    "billid": row["billid"],
                },
            )

print("✅ All nodes and relationships created successfully.")

# Embeddings

model = SentenceTransformer("all-MiniLM-L6-v2")

# Choose representative text fields per label
embedding_targets = {
    # Construct a readable patient name and include email for better
    # semantic search
    "Patient": (
        patients_df,
        lambda r: (
            " ".join([
                str(r.get("firstname", "")),
                str(r.get("lastname", "")),
            ]).strip()
            or str(r.get("patientid", ""))
        ),
    ),
    # Doctor full name + specialization
    "Doctor": (
        doctors_df,
        lambda r: (
            " ".join([
                str(r.get("firstname", "")),
                str(r.get("lastname", "")),
                "-",
                str(r.get("specialization", "")),
            ]).strip()
            or str(r.get("doctorid", ""))
        ),
    ),
    # Appointment summary (reason + status + date)
    "Appointment": (
        appointments_df,
        lambda r: (
            " | ".join([
                str(r.get("reasonforvisit", "")),
                str(r.get("status", "")),
                str(r.get("appointmentdate", "")),
            ]).strip()
            or str(r.get("appointmentid", ""))
        ),
    ),
    # Treatment type + description
    "Treatment": (
        treatments_df,
        lambda r: (
            " | ".join([
                str(r.get("treatmenttype", "")),
                str(r.get("description", "")),
            ]).strip()
            or str(r.get("treatmentid", ""))
        ),
    ),
    # Bill summary (status + method + amount + date)
    "Bill": (
        billing_df,
        lambda r: (
            " | ".join([
                str(r.get("paymentstatus", "")),
                str(r.get("paymentmethod", "")),
                str(r.get("amount", "")),
                str(r.get("billdate", "")),
            ]).strip()
            or str(r.get("billid", ""))
        ),
    ),
}


def set_embedding(tx, label, id_field, node_id, embedding):
    tx.run(
        f"""
        MATCH (n:{label} {{{id_field}: $node_id}})
        SET n.embedding = $embedding
        """,
        node_id=node_id,
        embedding=[float(x) for x in embedding],
    )

  
with driver.session() as session:
    for label, (df, text_fn) in embedding_targets.items():
        # Determine id field by label
        id_field = {
            "Patient": "patientid",
            "Doctor": "doctorid",
            "Appointment": "appointmentid",
            "Treatment": "treatmentid",
            "Bill": "billid",
        }[label]

        # Drop rows without ids
        df_valid = df[df[id_field].notnull()].copy()
        if df_valid.empty:
            print(f"Skipping {label}: no valid ids.")
            continue

        # Build texts with cleaning
        texts = []
        ids = []
        for _, r in df_valid.iterrows():
            text = str(text_fn(r)).strip()
            if text:
                texts.append(text)
                ids.append(r[id_field])

        if not texts:
            print(f"Skipping {label}: no valid text for embeddings.")
            continue

        embeddings = model.encode(
            texts,
            batch_size=64,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )

        for node_id, emb in zip(ids, embeddings):
            session.execute_write(set_embedding, label, id_field, node_id, emb)

        print(f"Uploaded embeddings for {label}")

driver.close()
print("✅ All embeddings uploaded to Neo4j.")'''


import os
import pandas as pd
from dotenv import load_dotenv, find_dotenv
from neo4j import GraphDatabase
from sentence_transformers import SentenceTransformer

# Load .env
load_dotenv(find_dotenv())

# Neo4j connection
NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))

# Load CSVs (update base_path as needed)
base_path = "./"

patients_df = pd.read_csv(f"{base_path}/patients.csv")
doctors_df = pd.read_csv(f"{base_path}/doctors.csv")
appointments_df = pd.read_csv(f"{base_path}/appointments.csv")
treatments_df = pd.read_csv(f"{base_path}/treatments.csv")
billing_df = pd.read_csv(f"{base_path}/billing.csv")

# Normalize column names
dfs = [patients_df, doctors_df, appointments_df, treatments_df, billing_df]
for df in dfs:
    df.columns = df.columns.astype(str).str.strip().str.lower().str.replace(" ", "_")

# Harmonize ID column names (rename once so all downstream uses are consistent)
def rename_if_exists(df, mapping):
    to_apply = {k: v for k, v in mapping.items() if k in df.columns}
    if to_apply:
        df.rename(columns=to_apply, inplace=True)

rename_if_exists(patients_df, {"patientid": "patient_id"})
rename_if_exists(doctors_df, {"doctorid": "doctor_id"})
rename_if_exists(appointments_df, {"appointmentid": "appointment_id"})
rename_if_exists(treatments_df, {"treatmentid": "treatment_id", "appointmentid": "appointment_id"})
rename_if_exists(billing_df, {"billid": "bill_id", "patientid": "patient_id"})

# Coerce and clean key date/time fields
# Patients: registrationdate, dateofbirth -> registration_date, date_of_birth (optional to rename)
rename_if_exists(patients_df, {"registrationdate": "registration_date", "dateofbirth": "date_of_birth"})
if "registration_date" in patients_df.columns:
    patients_df["registration_date"] = pd.to_datetime(patients_df["registration_date"], errors="coerce").dt.date
if "date_of_birth" in patients_df.columns:
    patients_df["date_of_birth"] = pd.to_datetime(patients_df["date_of_birth"], errors="coerce").dt.date

# Appointments: appointment_date, appointment_time
if "appointment_date" in appointments_df.columns:
    appointments_df["appointment_date"] = pd.to_datetime(appointments_df["appointment_date"], errors="coerce").dt.date
if "appointment_time" in appointments_df.columns:
    def normalize_time(val):
        s = str(val)
        if ":" in s:
            parts = s.split(":")
            if len(parts) == 3:
                h, m, sec = parts
                return f"{str(h).zfill(2)}:{str(m).zfill(2)}:{str(sec).zfill(2)}"
        digits = "".join(ch for ch in s if ch.isdigit())
        if len(digits) == 6:
            return f"{digits[0:2]}:{digits[2:4]}:{digits[4:6]}"
        return None
    appointments_df["appointment_time"] = appointments_df["appointment_time"].apply(normalize_time)

# Treatments: treatment_date, cost
rename_if_exists(treatments_df, {"treatmentdate": "treatment_date"})
if "treatment_date" in treatments_df.columns:
    treatments_df["treatment_date"] = pd.to_datetime(treatments_df["treatment_date"], errors="coerce").dt.date
if "cost" in treatments_df.columns:
    treatments_df["cost"] = pd.to_numeric(treatments_df["cost"], errors="coerce")

# Billing: bill_date, amount
rename_if_exists(billing_df, {"billdate": "bill_date"})
if "bill_date" in billing_df.columns:
    billing_df["bill_date"] = pd.to_datetime(billing_df["bill_date"], errors="coerce").dt.date
if "amount" in billing_df.columns:
    billing_df["amount"] = pd.to_numeric(billing_df["amount"], errors="coerce")

# Cypher helpers
def create_node(tx, label, id_key, row):
    props = {k: (None if pd.isna(v) else v) for k, v in dict(row).items()}
    props = {k: v for k, v in props.items() if v is not None}
    tx.run(f"MERGE (n:{label} {{{id_key}: $id}}) SET n += $props", id=row[id_key], props=props)

def create_relationship(tx, query, params):
    tx.run(query, **params)

# Step 1: Create nodes and relationships
with driver.session() as session:
    # Patients
    if "patient_id" not in patients_df.columns:
        raise KeyError(f"Expected 'patient_id' in patients_df, found {patients_df.columns.tolist()}")
    for _, row in patients_df.iterrows():
        session.execute_write(create_node, "Patient", "patient_id", row)

    # Doctors
    if "doctor_id" not in doctors_df.columns:
        raise KeyError(f"Expected 'doctor_id' in doctors_df, found {doctors_df.columns.tolist()}")
    for _, row in doctors_df.iterrows():
        session.execute_write(create_node, "Doctor", "doctor_id", row)

    # Appointments
    if "appointment_id" not in appointments_df.columns:
        raise KeyError(f"Expected 'appointment_id' in appointments_df, found {appointments_df.columns.tolist()}")
    for _, row in appointments_df.iterrows():
        session.execute_write(create_node, "Appointment", "appointment_id", row)
        # Patient -> Appointment
        pid = row.get("patient_id")
        if pd.notna(pid):
            session.execute_write(
                create_relationship,
                """
                MATCH (p:Patient {patient_id: $patient_id})
                MATCH (a:Appointment {appointment_id: $appointment_id})
                MERGE (p)-[:HAS_APPOINTMENT]->(a)
                """,
                {"patient_id": pid, "appointment_id": row["appointment_id"]},
            )
        # Doctor -> Appointment
        did = row.get("doctor_id")
        if pd.notna(did):
            session.execute_write(
                create_relationship,
                """
                MATCH (d:Doctor {doctor_id: $doctor_id})
                MATCH (a:Appointment {appointment_id: $appointment_id})
                MERGE (d)-[:PERFORMED_BY]->(a)
                """,
                {"doctor_id": did, "appointment_id": row["appointment_id"]},
            )

    # Treatments (Appointment -> Treatment)
    if "treatment_id" not in treatments_df.columns:
        raise KeyError(f"Expected 'treatment_id' in treatments_df, found {treatments_df.columns.tolist()}")
    for _, row in treatments_df.iterrows():
        session.execute_write(create_node, "Treatment", "treatment_id", row)
        ap_id = row.get("appointment_id")
        if pd.notna(ap_id):
            session.execute_write(
                create_relationship,
                """
                MATCH (a:Appointment {appointment_id: $appointment_id})
                MATCH (t:Treatment {treatment_id: $treatment_id})
                MERGE (a)-[:HAS_TREATMENT]->(t)
                """,
                {"appointment_id": ap_id, "treatment_id": row["treatment_id"]},
            )

    # Billing (Patient -> Bill, Treatment -> Bill)
    if "bill_id" not in billing_df.columns:
        raise KeyError(f"Expected 'bill_id' in billing_df, found {billing_df.columns.tolist()}")
    for _, row in billing_df.iterrows():
        session.execute_write(create_node, "Bill", "bill_id", row)
        pid = row.get("patient_id")
        if pd.notna(pid):
            session.execute_write(
                create_relationship,
                """
                MATCH (p:Patient {patient_id: $patient_id})
                MATCH (b:Bill {bill_id: $bill_id})
                MERGE (p)-[:HAS_BILL]->(b)
                """,
                {"patient_id": pid, "bill_id": row["bill_id"]},
            )
        tid = row.get("treatment_id")
        if pd.notna(tid):
            session.execute_write(
                create_relationship,
                """
                MATCH (t:Treatment {treatment_id: $treatment_id})
                MATCH (b:Bill {bill_id: $bill_id})
                MERGE (t)-[:BILLED_AS]->(b)
                """,
                {"treatment_id": tid, "bill_id": row["bill_id"]},
            )

print("All nodes and relationships created successfully.")

# Step 2: Embedding
model = SentenceTransformer("all-MiniLM-L6-v2")

# Step 3: Build readable text per label
def text_patient(r):
    parts = [str(r.get("firstname", "")).strip(), str(r.get("lastname", "")).strip()]
    txt = " ".join([p for p in parts if p]).strip()
    return txt or str(r.get("patient_id", ""))

def text_doctor(r):
    parts = [str(r.get("firstname", "")).strip(), str(r.get("lastname", "")).strip(), "-", str(r.get("specialization", "")).strip()]
    txt = " ".join([p for p in parts if p]).strip(" -")
    return txt or str(r.get("doctor_id", ""))

def text_appointment(r):
    parts = [str(r.get("reason_for_visit", r.get("reasonforvisit", ""))).strip(),
             str(r.get("status", "")).strip(),
             str(r.get("appointment_date", r.get("appointmentdate", ""))).strip()]
    txt = " | ".join([p for p in parts if p])
    return txt or str(r.get("appointment_id", ""))

def text_treatment(r):
    parts = [str(r.get("treatmenttype", "")).strip(), str(r.get("description", "")).strip()]
    txt = " | ".join([p for p in parts if p])
    return txt or str(r.get("treatment_id", ""))

def text_bill(r):
    parts = [str(r.get("paymentstatus", "")).strip(),
             str(r.get("paymentmethod", "")).strip(),
             str(r.get("amount", "")).strip(),
             str(r.get("bill_date", r.get("billdate", ""))).strip()]
    txt = " | ".join([p for p in parts if p])
    return txt or str(r.get("bill_id", ""))

label_id_fields = {
    "Patient": "patient_id",
    "Doctor": "doctor_id",
    "Appointment": "appointment_id",
    "Treatment": "treatment_id",
    "Bill": "bill_id",
}
label_text_fns = {
    "Patient": text_patient,
    "Doctor": text_doctor,
    "Appointment": text_appointment,
    "Treatment": text_treatment,
    "Bill": text_bill,
}

def set_embedding(tx, label, id_field, node_id, embedding):
    tx.run(
        f"""
        MATCH (n:{label} {{{id_field}: $node_id}})
        SET n.embedding = $embedding
        """,
        node_id=node_id,
        embedding=[float(x) for x in embedding],
    )

# Step 4: Clean and upload embeddings
with driver.session() as session:
    for label, df in [("Patient", patients_df), ("Doctor", doctors_df), ("Appointment", appointments_df), ("Treatment", treatments_df), ("Bill", billing_df)]:
        id_field = label_id_fields[label]
        text_fn = label_text_fns[label]

        df_valid = df[df[id_field].notnull()].copy()
        if df_valid.empty:
            print(f"Skipping {label}: no valid '{id_field}' entries.")
            continue

        texts, ids = [], []
        for _, r in df_valid.iterrows():
            txt = text_fn(r)
            if txt and str(txt).strip():
                texts.append(str(txt))
                ids.append(r[id_field])

        if not texts:
            print(f"Skipping {label}: no valid text.")
            continue

        embeddings = model.encode(texts)
        for node_id, emb in zip(ids, embeddings):
            session.execute_write(set_embedding, label, id_field, node_id, emb)

        print(f"Uploaded embeddings for {label}")

driver.close()
print("All embeddings uploaded to Neo4j.")

