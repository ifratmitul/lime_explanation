import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.applications import InceptionV3, ResNet50
from tensorflow.keras.applications.inception_v3 import preprocess_input as inception_preprocess
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet_preprocess
from tensorflow.keras.preprocessing.image import array_to_img
from tensorflow.image import resize
from lime import lime_image
from skimage.segmentation import mark_boundaries
from tensorflow.keras.models import load_model


(x_train, y_train), (x_test, y_test) = cifar10.load_data()
images = x_test  # Pick 5 test images

# Load the existing model
original_model = load_model('/kaggle/input/baseline_model/keras/default/1/main_model_no_stopped.h5')


def prepare_images(images, model_name='inception'):
    resized = np.array([
        resize(img, (299, 299)) if model_name == 'inception' 
        else resize(img, (224, 224)) 
        for img in images
    ])
    if model_name == 'inception':
        return inception_preprocess(resized)
    else:
        return resnet_preprocess(resized)

def predict_inception(imgs):
    return inception_model.predict(prepare_images(imgs, 'inception'))

def predict_resnet(imgs):
    return resnet_model.predict(prepare_images(imgs, 'resnet'))

# Baseline model prediction function (expects 32x32 input, normalized)
def predict_baseline(imgs):
    imgs = np.array(imgs) / 255.0  # normalize to [0, 1]
    return original_model.predict(imgs)

def get_all_feature_heatmap(ind, explnation):
    dict_heatmap = dict(explnation.local_exp[ind])
    heatmap = np.vectorize(dict_heatmap.get)(explnation.segments)
    return heatmap

def get_top_k_feature_heatmap(label_incep, explanation, num_features):
    top_features = explanation.local_exp[label_incep][:3]
    dict_heatmap_2 = dict(top_features)

    # Set zero for superpixels not in top_features
    heatmap_2 = np.zeros_like(explanation.segments, dtype=float)
    for seg_val, weight in dict_heatmap_2.items():
        heatmap_2[explanation.segments == seg_val] = weight

    return heatmap_2, dict_heatmap_2
    
inception_images = prepare_images(images, model_name='inception')
resnet_images = prepare_images(images, model_name='resnet')


inception_model = InceptionV3(weights='imagenet')
resnet_model = ResNet50(weights='imagenet')


# Initialize LIME explainer
explainer = lime_image.LimeImageExplainer()

# Select indices of 3 different CIFAR-10 images
image_indices = [10, 7617, 12]
top_k_feature =  3

for idx in image_indices:
    original_image = images[idx].astype('double')

    # InceptionV3
    explanation_incep = explainer.explain_instance(
        original_image,
        predict_inception,
        top_labels=5,
        hide_color=128,
        num_samples=1000
    )
    label_incep = explanation_incep.top_labels[0]
    temp_incep, mask_incep = explanation_incep.get_image_and_mask(
        label_incep, positive_only=True, num_features=top_k_feature, hide_rest=True
    )

    # ResNet50
    explanation_resnet = explainer.explain_instance(
        original_image,
        predict_resnet,
        top_labels=5,
        hide_color=128,
        num_samples=1000
    )
    label_resnet = explanation_resnet.top_labels[0]
    temp_resnet, mask_resnet = explanation_resnet.get_image_and_mask(
        label_resnet, positive_only=True, num_features=top_k_feature, hide_rest=True
    )

    # Baseline model
    explanation_baseline = explainer.explain_instance(
        original_image,
        predict_baseline,
        top_labels=5,
        hide_color=128,
        num_samples=1000
    )
    label_baseline = explanation_baseline.top_labels[0]
    temp_baseline, mask_baseline = explanation_baseline.get_image_and_mask(
        label_baseline, positive_only=True, num_features=top_k_feature, hide_rest=True
    )

    inception_heatmap = get_all_feature_heatmap(ind=label_incep, explnation=explanation_incep)
    inception_heatmap_top_features, dict_heatmap_2_incep = get_top_k_feature_heatmap(label_incep=label_incep, explanation=explanation_incep, num_features=top_k_feature)

    baseline_heatmap = get_all_feature_heatmap(ind=label_baseline, explnation=explanation_baseline)
    baseline_heatmap_top_features, dict_heatmap_2_base = get_top_k_feature_heatmap(label_incep=label_baseline, explanation=explanation_baseline, num_features=top_k_feature)
    
    fig, axs = plt.subplots(2, 3, figsize=(8.5, 5))
    fig.subplots_adjust(wspace=0.3, hspace=0.3)
    
    # -------- Inception Row --------
    axs[0, 0].imshow(original_image.astype('uint8'))
    axs[0, 0].set_title("Original CIFAR-10 Image")
    axs[0, 0].axis('off')
    
    axs[0, 1].imshow(temp_incep / 255.0)
    axs[0, 1].set_title("LIME (InceptionV3)")
    axs[0, 1].axis('off')
    
    max_abs_incep_top = np.max(np.abs(list(dict_heatmap_2_incep.values())))
    img_top_incep = axs[0, 2].imshow(inception_heatmap_top_features, cmap='RdBu_r', 
                                     vmin=-max_abs_incep_top, vmax=max_abs_incep_top)
    axs[0, 2].set_title("Heatmap")
    axs[0, 2].axis('off')
    fig.colorbar(img_top_incep, ax=axs[0, 2], fraction=0.046, pad=0.04)
    
    # img_all_incep = axs[0, 3].imshow(inception_heatmap, cmap='RdBu_r', 
    #                                  vmin=-inception_heatmap.max(), vmax=inception_heatmap.max())
    # axs[0, 3].set_title("All Features Heatmap (InceptionV3)")
    # axs[0, 3].axis('off')
    # fig.colorbar(img_all_incep, ax=axs[0, 3], fraction=0.046, pad=0.04)
    
    # -------- Baseline Row --------
    axs[1, 0].imshow(original_image.astype('uint8'))
    axs[1, 0].set_title("Original CIFAR-10 Image")
    axs[1, 0].axis('off')
    
    axs[1, 1].imshow(temp_baseline / 255.0)
    axs[1, 1].set_title("LIME (Baseline)")
    axs[1, 1].axis('off')
    
    max_abs_base_top = np.max(np.abs(list(dict_heatmap_2_base.values())))
    img_top_base = axs[1, 2].imshow(baseline_heatmap_top_features, cmap='RdBu_r', 
                                    vmin=-max_abs_base_top, vmax=max_abs_base_top)
    axs[1, 2].set_title("Heatmap")
    axs[1, 2].axis('off')
    fig.colorbar(img_top_base, ax=axs[1, 2], fraction=0.046, pad=0.04)
    
    # img_all_base = axs[1, 3].imshow(baseline_heatmap, cmap='RdBu_r', 
    #                                 vmin=-baseline_heatmap.max(), vmax=baseline_heatmap.max())
    # axs[1, 3].set_title("All Features Heatmap (Baseline)")
    # axs[1, 3].axis('off')
    # fig.colorbar(img_all_base, ax=axs[1, 3], fraction=0.046, pad=0.04)
    
    # Final adjustments
    plt.tight_layout()
    plt.savefig(f"LIME_heatmap_compare_{idx}.pdf", dpi=300, format='pdf')
    plt.show()
