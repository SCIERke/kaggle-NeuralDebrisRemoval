from visualization.channel_diagnoise import channel_diagnoise_visualization
from matplotlib import pyplot as plt
import numpy as np

if __name__ == "__main__":
    mean_of_channel, scores= channel_diagnoise_visualization()
    
    k = 10
    vals, idx = mean_of_channel.abs().topk(k)
    
    signed = mean_of_channel[idx]

    for channel_id, channel_score in zip(idx, signed):
        print(f"channel {channel_id.item()}: {channel_score.item():+.3f}")

    fig, ax = plt.subplots()
    im = ax.imshow(scores.detach().numpy() ,cmap='viridis')

    ax.set_title("Channel heatmap through 20 images")
    plt.tight_layout()
    plt.show()
    # plt.hist(mean_of_channel.detach().numpy(), bins=50)
    #
    # plt.title("Mean of all NN channel")
    # plt.xlabel("score_value")
    # plt.ylabel("channels")
    # plt.savefig('output/mean_of_channel.png')
    # plt.show()

