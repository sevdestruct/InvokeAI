import type { ComboboxOnChange, ComboboxOption } from '@invoke-ai/ui-library';
import { Combobox, FormControl, FormLabel } from '@invoke-ai/ui-library';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { InformationalPopover } from 'common/components/InformationalPopover/InformationalPopover';
import { selectAccelerationMode, setAccelerationMode } from 'features/controlLayers/store/paramsSlice';
import type { ParamsState } from 'features/controlLayers/store/types';
import { memo, useCallback, useMemo } from 'react';
import { useTranslation } from 'react-i18next';

const isValidAccelerationMode = (value: string | undefined): value is ParamsState['accelerationMode'] => {
  return value === 'off' || value === 'balanced' || value === 'max';
};

export const ParamAccelerationMode = memo(() => {
  const dispatch = useAppDispatch();
  const { t } = useTranslation();
  const accelerationMode = useAppSelector(selectAccelerationMode);

  const options = useMemo<ComboboxOption[]>(
    () => [
      { value: 'off', label: t('parameters.accelerationModeOff') },
      { value: 'balanced', label: t('parameters.accelerationModeBalanced') },
      { value: 'max', label: t('parameters.accelerationModeMax') },
    ],
    [t]
  );

  const value = useMemo(() => options.find((o) => o.value === accelerationMode), [options, accelerationMode]);

  const onChange = useCallback<ComboboxOnChange>(
    (v) => {
      if (!isValidAccelerationMode(v?.value)) {
        return;
      }
      dispatch(setAccelerationMode(v.value));
    },
    [dispatch]
  );

  return (
    <FormControl minW={0} flexGrow={1} gap={2}>
      <InformationalPopover feature="accelerationMode">
        <FormLabel m={0}>{t('parameters.accelerationMode')}</FormLabel>
      </InformationalPopover>
      <Combobox value={value} options={options} onChange={onChange} />
    </FormControl>
  );
});

ParamAccelerationMode.displayName = 'ParamAccelerationMode';
