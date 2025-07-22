import sys
import os
sys.path.append(os.path.abspath(os.path.dirname(__file__) + "/.."))

import joblib
import numpy as np
from models.scibert_embedder import SciBERTEmbedder as Task1Embedder
from scibert_embedder import SciBERTEmbedder as Task2Embedder

def cosine_similarity(a, b):
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    return np.dot(a, b)

def predict_publishability(text):
    model = joblib.load("models\scibert_small_classifier.pkl")
    embedder = Task1Embedder()
    emb = embedder.get_embeddings([text])
    X = model["scaler"].transform(emb)
    pred = model["classifier"].predict(X)[0]
    return pred

def predict_conference_and_rationale(text):
    model = joblib.load("model2/scibert_recommender.pkl")
    embedder = Task2Embedder()
    emb = embedder.get_embeddings([text])
    X = model["scaler"].transform(emb)
    pred = model["classifier"].predict(X)[0]

    # Load reference papers
    import os
    TEXT_DIR = "cleaned_texts"
    references = []
    for fname in os.listdir(TEXT_DIR):
        if fname.endswith(".txt"):
            with open(os.path.join(TEXT_DIR, fname), "r", encoding="utf-8") as f:
                content = f.read().strip()
                if content:
                    references.append((fname, content))

    texts = [t for _, t in references]
    ref_embs = embedder.get_embeddings(texts)
    sims = [cosine_similarity(emb[0], ref) for ref in ref_embs]
    top_k = np.argsort(sims)[-3:][::-1]
    top_refs = [(references[i][0], sims[i]) for i in top_k]

    rationale = f"The paper aligns with {pred} due to methodological and topical similarity with papers such as "
    rationale += ", ".join([f"{name.split('.')[0]}" for name, _ in top_refs])
    rationale += ". It shows relevance in themes common to " + pred + ", such as advanced experimentation and quality results."
    if len(rationale) > 100:
        rationale = rationale[:97] + "..."

    return pred, rationale

def full_pipeline(text):
    is_publishable = predict_publishability(text)
    if is_publishable == "non-publishable":
        return {
            "publishable": False,
            "message": "The paper is unlikely to be publishable based on current content quality."
        }
    
    conference, rationale = predict_conference_and_rationale(text)
    return {
        "publishable": True,
        "conference": conference,
        "rationale": rationale
    }
if __name__ == "__main__":
    sample_text = "Detecting Medication Usage in Parkinson s Disease Through Multi-modal Indoor Positioning: A Pilot Study in a Naturalistic Environment Abstract Parkinson s disease (PD) is a progressive neurodegenerative disorder that leads to motor symptoms, including gait impairment. The effectiveness of levodopa therapy, a common treatment for PD, can fluctuate, causing periods of improved mobility (on state) and periods where symptoms re-emerge (off state). These fluctuations impact gait speed and increase in severity as the disease progresses. This paper proposes a transformer-based method that uses both Received Signal Strength Indicator (RSSI) and accelerometer data from wearable devices to enhance indoor localization accuracy. A secondary goal is to determine if indoor localization, particularly in-home gait speed features (like the time to walk between rooms), can be used to identify motor fluctuations by detecting if a person with PD is taking their levodopa medication or not. The method is evaluated using a real-world dataset collected in a free-living setting, where movements are varied and unstructured. Twenty-four participants, living in pairs (one with PD and one control), resided in a sensor-equipped smart home for five days. The results show that the proposed network surpasses other methods for indoor localization. The evaluation of the secondary goal reveals that accurate room-level localization, when converted into in-home gait speed features, can accurately predict whether a PD participant is taking their medication or not. 1 Introduction Parkinson s disease (PD) is a debilitating neurodegenerative condition that affects approximately 6 million individuals globally. It manifests through various motor symptoms, including bradykinesia (slowness of movement), rigidity, and gait impairment. A common complication associated with levodopa, the primary medication for PD, is the emergence of motor fluctuations that are linked to medication timing. Initially, patients experience a consistent and extended therapeutic effect when starting levodopa. However, as the disease advances, a significant portion of patients begin to experience wearing off of their medication before the next scheduled dose, resulting in the reappearance of parkinsonian symptoms, such as slowed gait. These fluctuations in symptoms negatively impact patients  quality of life and often necessitate adjustments to their medication regimen. The severity of motor symptoms can escalate to the point where they impede an individual s ability to walk and move within their own home. Consequently, individuals may be inclined to remain confined to a single room, and when they do move, they may require more time to transition between rooms. These observations could potentially be used to identify periods when PD patients are experiencing motor fluctuations related to their medication being in an ON or OFF state, thereby providing valuable information to both clinicians and patients. A sensitive and accurate ecologically-validated biomarker for PD progression is currently unavailable, which has contributed to failures in clinical trials for neuroprotective therapies in PD. Gait parameters are sensitive to disease progression in unmedicated early-stage PD and show promise as markers of disease progression, making measuring gait parameters potentially useful in clinical trials of disease-modifying interventions. Clinical evaluations of PD are typically conducted in artificial clinic or laboratory settings, which only capture a limited view of an individual s motor function. Continuous monitoring could capture symptom progression, including motor fluctuations, and sensitively quantify them over time. While PD symptoms, including gait and balance parameters, can be measured continuously at home using wearable devices containing inertial motor units (IMUs) or smartphones, this data does not show the context in which the measurements are taken. Determining a person s location within a home (indoor localization) could provide valuable contextual information for interpreting PD symptoms. For instance, symptoms like freezing of gait and turning in gait vary depending on the environment, so knowing a person s location could help predict such symptoms or interpret their severity. Additionally, understanding how much time someone spends alone or with others in a room is a step towards understanding their social participation, which impacts quality of life in PD. Localization could also provide valuable information in the measurement of other behaviors such as non-motor symptoms like urinary function (e.g., how many times someone visits the toilet room overnight). IoT-based platforms with sensors capturing various modalities of data, combined with machine learning, can be used for unobtrusive and continuous indoor localization in home environments. Many of these techniques utilize radio-frequency signals, specifically the Received Signal Strength Indication (RSSI), emitted by wearables and measured at access points (AP) throughout a home. These signals estimate the user s position based on perceived signal strength, creating radio-map features for each room. To improve localization accuracy, accelerometer data from wearable devices, along with RSSI, can be used to distinguish different activities (e.g., walking vs. standing). Since some activities are associated with specific rooms (e.g., stirring a pan on the stove is likely to occur in a kitchen), accelerometer data can enhance RSSI s ability to differentiate between adjacent rooms, an area where RSSI alone may be insufficient. The heterogeneity of PD, where symptoms and their severity vary between patients, poses a challenge for generalizing accelerometer data across different individuals. Severe symptoms, such as tremors, can introduce bias and accumulated errors in accelerometer data, particularly when collected from wrist-worn devices, which are a common and well-accepted placement location. Naively combining accelerometer data with RSSI may degrade indoor localization performance due to varying tremor levels in the acceleration signal. This work makes two primary contributions to address these challenges. (1) We detail the use of RSSI, augmented by accelerometer data, to achieve room-level localization. Our proposed network intelligently selects accelerometer features that can enhance RSSI performance in indoor localization. To rigorously assess our method, we utilize a free-living dataset (where individuals live without external intervention) developed by our group, encompassing diverse and unstructured movements as expected in real-world scenarios. Evaluation on this dataset, including individuals with and without PD, demonstrates that our network outperforms other methods across all cross-validation categories. (2) We demonstrate how accurate room-level localization predictions can be transformed into in-home gait speed biomarkers (e.g., number of room-to-room transitions, room-to-room transition duration). These biomarkers can effectively classify the OFF or ON medication state of a PD patient from this pilot study data. 2 Related Work Extensive research has utilized home-based passive sensing systems to evaluate how the activities and behavior of individuals with neurological conditions, primarily cognitive dysfunction, change over time. However, there is limited work assessing room use in the home setting in people with Parkinson s. Gait quantification using wearables or smartphones is an area where a significant amount of work has been done. Cameras can also detect parkinsonian gait and some gait features, including step length and average walking speed. Time-of-flight devices, which measure distances between the subject and the camera, have been used to assess medication adherence through gait analysis. From free-living data, one approach to gait and room use evaluation in home settings is by emitting and detecting radio waves to non-invasively track movement. Gait analysis using radio wave technology shows promise to track disease progression, severity, and medication response. However, this approach cannot identify who is doing the movement and also suffers from technical issues when the radio waves are occluded by another object. Much of the work done so far using video to track PD symptoms has focused on the performance of structured clinical rating scales during telemedicine consultations as opposed to naturalistic behavior, and there have been some privacy concerns around the use of video data at home. RSSI data from wearable devices is a type of data with fewer privacy concerns; it can be measured continuously and unobtrusively over long periods to capture real-world function and behavior in a privacy-friendly way. In indoor localization, fingerprinting using RSSI is the typical technique used to estimate the wearable (user) location by using signal strength data representing a coarse and noisy estimate of the distance from the wearable to the access point. RSSI signals are not stable; they fluctuate randomly due to shadowing, fading, and multi-path effects. However, many techniques have been proposed in recent years to tackle these fluctuations and indirectly improve localization accuracy. Some works utilize deep neural networks (DNN) to generate coarse positioning estimates from RSSI signals, which are then refined by a hidden Markov model (HMM) to produce a final location estimate. Other works try to utilize a time series of RSSI data and exploit the temporal connections within each access point to estimate room-level position. A CNN is used to build localization models to further leverage the temporal dependencies across time-series readings. It has been suggested that we cannot rely on RSSI alone for indoor localization in home environments for PD subjects due to shadowing rooms with tight separation. Some researchers combine RSSI signals and inertial measurement unit (IMU) data to test the viability of leveraging other sensors in aiding the positioning system to produce a more accurate location estimate. Classic machine learning approaches such as Random Forest (RF), Artificial Neural Network (ANN), and k-Nearest Neighbor (k-NN) are tested, and the result shows that the RF outperforms other methods in tracking a person in indoor environments. Others combine smartphone IMU sensor data and Wi-Fi-received signal strength indication (RSSI) measurements to estimate the exact location (in Euclidean position X, Y) of a person in indoor environments. The proposed sensor fusion framework uses location fingerprinting in combination with a pedestrian dead reckoning (PDR) algorithm to reduce positioning errors. Looking at this multi-modality classification/regression problem from a time series perspective, there has been a lot of exploration in tackling a problem where each modality can be categorized as multivariate time series data. LSTM and attention layers are often used in parallel to directly transform raw multivariate time series data into a low-dimensional feature representation for each modality. Later, various processes are done to further extract correlations across modalities through the use of various layers (e.g., concatenation, CNN layer, transformer, self-attention)."
    result = full_pipeline(sample_text)
    print(result)
