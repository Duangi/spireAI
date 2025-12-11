import os
import time
import socket
import queue
import threading
import torch
import wandb
import itertools
from multiprocessing.managers import BaseManager
from datetime import datetime

# Project Imports
from spirecomm.ai.dqn import DQNAgent
from spirecomm.ai.dqn_core.algorithm import SpireAgent
from spirecomm.ai.dqn_core.model import SpireConfig
from spirecomm.ai.dqn_core.wandb_logger import WandbLogger
from spirecomm.spire.character import PlayerClass
from spirecomm.communication.coordinator import Coordinator
from spirecomm.utils.path import get_root_dir
from spirecomm.ai.absolute_logger import AbsoluteLogger, LogType

# ==========================================
# 1. Configuration & Constants
# ==========================================
LEARNER_PORT = 6000
AUTH_KEY = b'spire_secret'

# --- Training Hyperparameters ---
# Note: In distributed training, the Learner counts optimization steps, not episodes.
# We convert episode-based frequencies to approximate step counts.
# Assuming ~500 steps per episode (very rough estimate).
STEPS_PER_EPISODE_EST = 500 

TARGET_UPDATE_FREQUENCY_EPISODES = 10
SAVE_MODEL_FREQUENCY_EPISODES = 10

# Convert to steps for the Learner
TARGET_UPDATE_STEPS = TARGET_UPDATE_FREQUENCY_EPISODES * STEPS_PER_EPISODE_EST
SAVE_MODEL_STEPS = SAVE_MODEL_FREQUENCY_EPISODES * STEPS_PER_EPISODE_EST

# --- Actor Configuration ---
# Which character to train?
PLAYER_CLASS_TO_TRAIN = PlayerClass.THE_SILENT
TRAIN_SINGLE_CLASS_MODE = True # If True, only train the above class. If False, cycle through all.
ASCENSION_LEVEL = 20

# ==========================================
# 2. IPC Manager
# ==========================================
class SharedStorage:
    """Wrapper to manage shared state safely via SyncManager"""
    def __init__(self):
        self.weights = {}
    
    def get_weights(self):
        return self.weights
    
    def update_weights(self, w):
        self.weights.update(w)

class ClientIdProvider:
    """Thread-safe counter for assigning Client IDs"""
    def __init__(self):
        self.lock = threading.Lock()
        self.next_id = 1
    
    def get_next_id(self):
        with self.lock:
            cid = self.next_id
            self.next_id += 1
            return cid

class SpireManager(BaseManager):
    pass

# Global containers for the Manager process
exp_queue = queue.Queue(maxsize=5000)
# We will use a shared object for weights instead of a raw dict
shared_storage = SharedStorage()
client_id_provider = ClientIdProvider()

def get_exp_queue():
    return exp_queue

def get_shared_storage():
    return shared_storage

def get_client_id_provider():
    return client_id_provider

SpireManager.register('get_exp_queue', callable=get_exp_queue)
SpireManager.register('get_shared_storage', callable=get_shared_storage)
SpireManager.register('get_client_id_provider', callable=get_client_id_provider)

# ==========================================
# 3. Helper Functions
# ==========================================
def is_learner_running(port):
    """Check if the Learner port is occupied."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(('127.0.0.1', port)) == 0

def setup_coordinator_for_agent(agent: DQNAgent) -> Coordinator:
    """Creates and registers Coordinator callbacks."""
    coord = Coordinator()
    coord.signal_ready()
    coord.register_command_error_callback(agent.handle_error)
    coord.register_state_change_callback(agent.get_next_action_in_game)
    coord.register_out_of_game_callback(agent.get_next_action_out_of_game)
    return coord

# ==========================================
# 4. Actor Agent (Subclass)
# ==========================================
class ActorAgent(DQNAgent):
    def __init__(self, shared_queue, shared_storage_proxy, logger=None, client_id="Client", *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.shared_queue = shared_queue
        self.shared_storage_proxy = shared_storage_proxy
        self.last_sync_time = 0
        self.sync_interval = 10 # Sync weights every 10 steps/calls
        self.logger = logger
        self.client_id = client_id
        self.step_counter = 0

        # Monkey-patch the internal algorithm's remember and train methods
        self.dqn_algorithm.remember = self.remote_remember
        self.dqn_algorithm.train = self.sync_weights_trigger

    def remote_remember(self, state, action, reward, next_state, done, reward_details=""):
        """Push experience to the shared queue instead of local memory."""
        try:
            # Ensure tensors are on CPU to be picklable and save bandwidth
            # SpireState objects usually contain CPU tensors by default from GameStateProcessor
            # but we double check if we need to move them.
            # Assuming state/next_state are SpireState objects.
            
            # We put the tuple into the queue
            # Use timeout to avoid infinite blocking if Server is dead
            self.shared_queue.put((state, action, reward, next_state, done, reward_details), timeout=5)
            
            self.step_counter += 1
            if self.logger and self.step_counter % 10 == 0:
                try:
                    q_size = self.shared_queue.qsize()
                except:
                    q_size = "?"
                self.logger.write(f"【{self.client_id}】Generated {self.step_counter} steps. Queue size: {q_size}\n")
                if hasattr(self.logger, 'file_handle') and self.logger.file_handle:
                    self.logger.file_handle.flush()

        except Exception as e:
            pass # Silent error

    def sync_weights_trigger(self):
        """Check and sync weights from shared memory."""
        # We don't want to sync every single step to avoid IPC overhead
        # But train() is called every step in DQNAgent.
        if time.time() - self.last_sync_time > 5: # Sync every 5 seconds
            self.sync_weights()
            self.last_sync_time = time.time()

    def sync_weights(self):
        try:
            # shared_storage_proxy is an AutoProxy to SharedStorage
            # We call get_weights() on it
            remote_dict = self.shared_storage_proxy.get_weights()
            if 'state_dict' in remote_dict:
                self.dqn_algorithm.policy_net.load_state_dict(remote_dict['state_dict'])
        except Exception as e:
            pass # Silent error

# ==========================================
# 5. Learner Logic (Server)
# ==========================================
def run_as_learner():
    # 1. Start Manager
    # Bind to 127.0.0.1 explicitly to avoid WinError 10049
    manager = SpireManager(address=('127.0.0.1', LEARNER_PORT), authkey=AUTH_KEY)
    manager.start()
    
    shared_q = manager.get_exp_queue()
    shared_s = manager.get_shared_storage()
    
    # 2. Init Agent & WandB
    config = SpireConfig()
    wandb_logger = WandbLogger(project_name="spire-ai-train", run_name=f"Learner_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    
    # Learner uses the standard SpireAgent (on GPU if available)
    agent = SpireAgent(config, wandb_logger=wandb_logger)
    agent.batch_size = 128 # Increase batch size for better GPU utilization
    
    # --- Logger Setup ---
    server_logger = AbsoluteLogger(LogType.STATE)
    server_logger.start_episode(filename_suffix="_Server")
    server_logger.write(f"【Server】Learner Process Started at {datetime.now()}\n")
    server_logger.file_handle.flush()

    # --- Load Latest Model if Available ---
    models_dir = os.path.join(get_root_dir(), "models")
    latest_episode = 0
    latest_model_path = None
    
    if os.path.exists(models_dir):
        # Look for dqn_learner_*.pth files
        model_files = [f for f in os.listdir(models_dir) if f.startswith("dqn_learner_") and f.endswith(".pth")]
        for f in model_files:
            try:
                # Extract update count from filename: dqn_learner_1000.pth
                update_num = int(f.split('_')[-1].split('.')[0])
                if update_num > latest_episode:
                    latest_episode = update_num
                    latest_model_path = os.path.join(models_dir, f)
            except ValueError:
                continue
    
    if latest_model_path:
        try:
            agent.load_model(latest_model_path)
            # If we loaded a model, we should probably set updates to latest_episode
            # But 'updates' variable is local here.
        except Exception:
            pass

    # Initial weights push
    cpu_dict = {k: v.cpu() for k, v in agent.policy_net.state_dict().items()}
    shared_s.update_weights({'state_dict': cpu_dict})
    
    
    updates = latest_episode
    
    try:
        while True:
            # A. Collect Data
            # Fetch a batch of experiences from the queue
            fetched_count = 0
            while not shared_q.empty() and fetched_count < 100:
                try:
                    transition = shared_q.get_nowait()
                    agent.remember(*transition)
                    fetched_count += 1
                except queue.Empty:
                    break
            
            if fetched_count == 0:
                time.sleep(0.1)
                continue
                
            # B. Train
            # Train as much as we can based on the data we have
            # Or maintain a ratio. For now, let's train once per batch of data or just loop
            if len(agent.memory) >= agent.batch_size:
                # Train multiple times to utilize GPU better and learn faster from collected data
                # But limit it to avoid overfitting on small buffer
                train_steps = 4 if len(agent.memory) > 1000 else 1
                
                for _ in range(train_steps):
                    agent.train()
                    updates += 1
                    
                    if updates % 10 == 0:
                        server_logger.write(f"【Server】Update {updates}: Trained batch. Memory size: {len(agent.memory)}\n")
                        server_logger.file_handle.flush()

                    # C. Sync Weights to Actors
                    if updates % 50 == 0:
                        cpu_dict = {k: v.cpu() for k, v in agent.policy_net.state_dict().items()}
                        shared_s.update_weights({'state_dict': cpu_dict})
                    
                    # D. Update Target Net
                    if updates % TARGET_UPDATE_STEPS == 0:
                        agent.update_target_net()
                    
                    # E. Save Model
                    if updates % SAVE_MODEL_STEPS == 0:
                        models_dir = os.path.join(get_root_dir(), "models")
                        os.makedirs(models_dir, exist_ok=True)
                        save_path = os.path.join(models_dir, f"dqn_learner_{updates}.pth")
                        agent.save_model(save_path)

    except KeyboardInterrupt:
        manager.shutdown()
        wandb_logger.finish()

# ==========================================
# 6. Actor Logic (Client)
# ==========================================
def run_as_actor():
    
    manager = SpireManager(address=('127.0.0.1', LEARNER_PORT), authkey=AUTH_KEY)
    try:
        manager.connect()
    except Exception as e:
        return

    shared_q = manager.get_exp_queue()
    shared_s = manager.get_shared_storage()
    
    # --- Get Client ID ---
    try:
        id_provider = manager.get_client_id_provider()
        client_num = id_provider.get_next_id()
        client_id = f"Client{client_num}"
    except Exception:
        # Fallback if something goes wrong
        client_id = f"Client_{os.getpid()}"

    # --- Logger Setup ---
    client_logger = AbsoluteLogger(LogType.STATE)
    client_logger.start_episode(filename_suffix=f"_{client_id}")
    client_logger.write(f"【{client_id}】Actor Process Started at {datetime.now()}\n")
    client_logger.file_handle.flush()
    
    # Actor runs on CPU to save GPU for Learner
    # We pass shared_q and shared_s to the agent
    agent = ActorAgent(shared_queue=shared_q, shared_storage_proxy=shared_s, logger=client_logger, client_id=client_id, play_mode=False)
    
    # Initial weight sync
    agent.sync_weights()
    
    # Setup Coordinator (Stdin/Stdout communication with Game)
    coordinator = setup_coordinator_for_agent(agent)
    
    # Register exit callback
    def on_exit():
        pass
    coordinator.register_on_exit_callback(on_exit)
    
    # Character Selection Logic
    if TRAIN_SINGLE_CLASS_MODE:
        player_class_cycle = itertools.cycle([PLAYER_CLASS_TO_TRAIN])
    else:
        player_class_cycle = itertools.cycle(PlayerClass)

    while True:
        chosen_class = next(player_class_cycle)
        agent.change_class(chosen_class)
        # Ensure config matches the character (though currently numeric_player_dim is constant 5)
        agent.dqn_algorithm.cfg.numeric_player_dim = 5 
        
        # Play one game
        # The agent's 'remember' method is patched to push to shared_q
        # The agent's 'train' method is patched to sync weights
        coordinator.play_one_game(chosen_class, ascension_level=ASCENSION_LEVEL)
        
        # After game, we can do a quick weight sync
        agent.sync_weights()

# ==========================================
# 7. Main Entry Point
# ==========================================
if __name__ == '__main__':
    # Set proxy if needed (copied from train.py)
    os.environ["HTTP_PROXY"] = "http://127.0.0.1:7890"
    os.environ["HTTPS_PROXY"] = "http://127.0.0.1:7890"
    # os.environ["WANDB_SILENT"] = "true" # Enable WandB output for debugging

    if is_learner_running(LEARNER_PORT):
        # Port occupied -> Learner is running -> I am an Actor
        try:
            run_as_actor()
        except Exception as e:
            raise e
            # Keep window open if it crashes immediately
            time.sleep(10)
    else:
        # Port free -> I am the Learner
        try:
            run_as_learner()
        except OSError:
            # Race condition: someone else took the port
            run_as_actor()
