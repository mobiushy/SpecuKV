import triton

print(triton.runtime.driver.active.get_current_target().arch)