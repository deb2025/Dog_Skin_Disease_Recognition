import streamlit as st
import tensorflow as tf
import numpy as np


#Tensorflow Model Prediction
def model_prediction(test_image):
    model = tf.keras.models.load_model("SkinDisease.weights.h5")
    image = tf.keras.preprocessing.image.load_img(test_image,target_size=(150,150))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr]) #convert single image to batch
    predictions = model.predict(input_arr)
    return np.argmax(predictions) #return index of max element

#Sidebar
st.sidebar.title("Dashboard")
app_mode = st.sidebar.selectbox("Select Page",["Home","About","Disease Recognition"])



#Main Page
if(app_mode=="Home"):
    st.header("DOG SKIN DISEASE RECOGNITION SYSTEM")
    image_path = "home_page.jpg"
    st.image(image_path,use_column_width=True)
    st.markdown("""
    # Welcome to the Dog Skin Disease Recognition System! 🐶🔍

    Our mission is to assist in identifying skin diseases in dogs efficiently. Upload an image of a dog's skin area, and our system will analyze it to detect any signs of diseases. Together, let's ensure our furry friends' skin health!

    ### How It Works
    1. **Upload Image:** Go to the **Skin Disease Recognition** page and upload an image of a dog's skin with suspected diseases.
    2. **Analysis:** Our system will process the image using advanced algorithms to identify potential skin diseases.
    3. **Results:** View the results and recommendations for further action.

    ### Why Choose Us?
    - **Accuracy:** Our system utilizes state-of-the-art machine learning techniques for precise disease detection.
    - **User-Friendly:** Simple and intuitive interface for seamless user experience.
    - **Fast and Efficient:** Receive results in seconds, allowing for quick decision-making.

    ### Get Started
    Click on the **Skin Disease Recognition** page in the sidebar to upload an image and experience the power of our Dog Skin Disease Recognition System!

    ### About Us
    Learn more about the project, our team, and our goals on the **About** page.

    """)

#About Project
elif(app_mode=="About"):
    st.header("About")
    image_path = "about.jpg"
    st.image(image_path,use_column_width=True)
    st.markdown("""
                #### About Team
                Some member of our team had diligently worked upon creating the dataset with extreme effort and other members trained the model and deployed it. 
                ### Disclaimer
                Please note that this is just merely an AI predicting skin disease through Image Recognition technique. We have built our model with the help of tensorflow library.We have only 4 classes or 4 disease that the model can predict roughly.
                ### Tech Stack Used---
                1. **Preprocessing Technique -** Offline augmentation , Image normalization
                2. **Model Structure -** Classic CNN
                3. **Model Input -** Image
                4. **Model Output -** Classification of Dog Dkin Diseases
                5. **Model Deployment and Frontend -** Streamlit 

                #### Content
                1. **Train** (373 images)
                2. **Test** (80 images)
                3. **Validation** (373 images)


                """)

#Prediction Page
elif(app_mode=="Disease Recognition"):
    st.header("Disease Recognition")
    test_image = st.file_uploader("Choose an Image:")
    if(st.button("Show Image")):
        st.image(test_image,width=4,use_column_width=True)
    #Predict button
    if(st.button("Predict")):
        st.snow()
        st.write("Our Prediction")
        result_index = model_prediction(test_image)
        #Reading Labels
        class_name = ['flea_allergy', 'hotspot', 'mange', 'ringworm']
        st.success("We are predicting it's a {}".format(class_name[result_index]))
        if(class_name[result_index]=='flea_allergy'):
            st.write("""
                 Flea allergy dermatitis (FAD) is a common allergic reaction in dogs caused by flea bites. Dogs with flea allergy are hypersensitive to flea saliva, which triggers an allergic response leading to intense itching, redness, inflammation, and skin lesions. Even a single flea bite can cause a significant reaction in allergic dogs.
                 ### Symptoms of flea allergy dermatitis in dogs may include:
                 1. **Intense itching:** Dogs may scratch, chew, or bite at their skin excessively, especially around the base of the tail, groin, belly, and back legs.
                 2. **Redness and inflammation:** Affected areas may become red, swollen, and tender.
                 3. **Hair loss:** Constant scratching and chewing can lead to patches of hair loss.
                 4. **Skin lesions:** In severe cases, dogs may develop hot spots or raw, oozing skin sores.
                ### Veterinarians typically recommend a multi-faceted approach to managing flea allergy dermatitis in dogs, which may include:
                1. **Flea control:** The primary treatment for flea allergy dermatitis is flea prevention. Veterinarians may recommend topical or oral flea preventatives that kill fleas and prevent infestations.
                2. **Antihistamines:** Some dogs may benefit from antihistamines to help alleviate itching and reduce the allergic response. Common antihistamines used in dogs include diphenhydramine and cetirizine, among others. However, the effectiveness can vary between individuals.
                3. **Steroids:** In cases of severe itching and inflammation, veterinarians may prescribe corticosteroids to quickly reduce inflammation and provide relief. However, long-term use of steroids may have side effects and should be used cautiously.
                4. **Topical treatments:** Medicated shampoos, sprays, or topical creams containing ingredients like oatmeal, aloe vera, or hydrocortisone can soothe irritated skin and help control itching.
                5. **Allergy testing and immunotherapy:** For dogs with severe or recurrent allergies, veterinarians may recommend allergy testing to identify specific allergens, followed by allergen-specific immunotherapy (allergy shots) to desensitize the dog's immune system to those allergens.
                ### ---------------But if the problem persists please contact a veterinarian as soon as possible-----------------
                ### -------THANKS FOR USING OUR SERVICE--------
 """)
        elif(class_name[result_index]=='hotspot'):
                  st.write("""
                 Hot spots, also known as acute moist dermatitis, are a common skin condition in dogs characterized by localized areas of inflammation, redness, and intense itching. These areas of irritated skin can develop rapidly and may be quite painful for the affected dog.
                 ### Symptoms of hot spots in dogs may include:
                 1. **Moist, red, and inflamed skin:** Hot spots typically appear as red, moist lesions on the dog's skin, often with matted fur around the affected area.
                 2. **Intense itching and discomfort:** Dogs with hot spots may obsessively lick, chew, or scratch at the affected area, exacerbating the inflammation and causing further irritation.
                 3. **Hair loss:** Due to constant licking and chewing, hot spots may result in patches of hair loss around the affected area.
                 4. **Oozing and crusting:** In severe cases, hot spots may ooze pus or serum and form crusts on the surface of the skin.
                ### Treatment of hot spots in dogs typically involves:
                1. **Trimming and cleaning the affected area:** The first step in treating a hot spot is to trim the hair around the lesion to allow air to reach the skin and facilitate drying. The area should then be gently cleaned with a mild antiseptic solution to remove debris and bacteria.
                2. **Topical medications:** Veterinarians may prescribe topical medications such as corticosteroid creams or sprays to reduce inflammation and soothe the skin. Additionally, topical antibiotics may be used to prevent or treat secondary bacterial infections.
                3. **E-collar (Elizabethan collar):** To prevent further trauma to the affected area from licking or chewing, dogs with hot spots may need to wear an Elizabethan collar to restrict access to the lesion.
                4. **Systemic medications:** In some cases, oral medications such as antibiotics or corticosteroids may be prescribed to address underlying infections or inflammation contributing to the hot spot.
                5. **Identifying and addressing underlying causes:** Hot spots can be triggered by various underlying factors, including allergies, flea infestations, skin infections, or underlying skin conditions. Identifying and addressing the underlying cause is essential to prevent recurrence.
                ### ---------------If the hot spot does not improve or if your dog develops additional lesions, it's important to consult a veterinarian for further evaluation and treatment-----------------
                ### -------THANKS FOR USING OUR SERVICE--------
 """)
        elif(class_name[result_index]=='mange'):
            st.write("""
                 Mange is a skin condition in dogs caused by mites, which are microscopic parasites. There are two primary types of mange in dogs: Sarcoptic mange (caused by Sarcoptes scabiei mites) and Demodectic mange (caused by Demodex canis mites). Mange can cause intense itching, hair loss, and skin irritation in affected dogs.
                 ### Symptoms of mange in dogs may include:
                 1. **Intense itching:** Dogs with mange often experience severe itching, which may lead to scratching, biting, and rubbing of the affected areas.
                 2. **Hair loss:** Mange mites burrow into the skin and hair follicles, causing hair loss, especially on the face, ears, elbows, and legs. 
                 3. **Skin lesions:** Infected areas may develop redness, inflammation, crusty patches, and sores.
                 4. **Thickened skin:** In chronic cases, the skin may become thickened and wrinkled.
                 5. **Secondary infections:** Constant scratching and open sores can lead to bacterial or fungal infections.
                ### Treatment for mange in dogs typically involves:
                1. **Medicated baths:** Veterinarians may prescribe medicated shampoos or dips containing ingredients like benzoyl peroxide, sulfur, or lime sulfur to kill mites and soothe the skin.
                2. **Topical medications:** Prescription ointments or creams containing antiparasitic medications may be applied directly to the affected areas to kill mites and reduce inflammation.
                3. **Oral medications:** In severe cases, veterinarians may prescribe oral medications such as ivermectin or milbemycin to kill mites systemically.
                4. **Antibiotics or antifungals:** If secondary infections are present, antibiotics or antifungal medications may be necessary to treat the infections.
                5. **Environmental management:** It's essential to thoroughly clean and disinfect the dog's environment to prevent reinfestation.
                ### ---------------If you suspect your dog has mange, it's crucial to consult a veterinarian for proper diagnosis and treatment-----------------
                ### -------THANKS FOR USING OUR SERVICE--------
 """)
        elif(class_name[result_index]=='ringworm'):
             st.write("""
                 Ringworm is a common fungal infection in dogs caused by various species of dermatophyte fungi, including Microsporum and Trichophyton. Despite its name, ringworm is not caused by worms but rather by fungi that invade the outer layers of the skin, hair, and sometimes nails. It is highly contagious and can spread between dogs, cats, and humans.
                 ### Symptoms of ringworm in dogs may include:
                 1. **Circular lesions:** Affected areas often appear as raised, red, and scaly patches with a clearer center, resembling a ring. However, not all lesions exhibit this classic ring-shaped appearance.
                 2. **Hair loss:** Infected hair follicles may become brittle and break, leading to patchy hair loss or bald spots.
                 3. **Itching:** Some dogs may experience mild to moderate itching or discomfort in the affected areas.
                 4. **Crusty or crusty lesions:** In severe cases, lesions may become crusty or develop pustules.
                ### Treatment for ringworm in dogs typically involves:
                1. **Antifungal medication:** Oral antifungal medications, such as griseofulvin, terbinafine, or itraconazole, are often prescribed to eliminate the fungal infection. Topical antifungal creams or shampoos may also be recommended for localized lesions.
                2. **Environmental decontamination:** Since ringworm spores can survive in the environment for an extended period, thorough cleaning of the dog's living areas and bedding is essential to prevent re-infection.
                3. **Isolation:** Infected dogs should be isolated from other pets and humans to prevent the spread of the infection.
                4. **Symptomatic treatment:** Veterinarians may prescribe medications to alleviate itching or discomfort associated with ringworm lesions.
                5. **Regular monitoring:** Close monitoring of the dog's progress and follow-up visits with the veterinarian are necessary to ensure the infection resolves completely.
                ### ---------------If you suspect your dog has ringworm, consult a veterinarian for proper diagnosis and treatment-----------------
                ### -------THANKS FOR USING OUR SERVICE--------
 """)
        else:
             st.write(""" 
                ### ------Please check your image and try again and now our model is in a very early stage so dont expect it to be fully perfect or fully responsive-------
                ### -------THANKS FOR USING OUR SERVICE--------
""")    
